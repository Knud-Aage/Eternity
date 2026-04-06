package dk.roleplay;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import static jcuda.driver.JCudaDriver.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class MasterSolverPBP implements Runnable {
    private final PieceInventory inventory;

    private final int[] flatBoard = new int[256];
    private final int[] bestBoard = new int[256];
    private final int[] flatResumeBoard = new int[256];
    private final boolean[] usedPhysicalPieces = new boolean[256];
    private final Random rnd = new Random();
    private int centerPhysicalIdx = -1;
    private int deepestPos = 0;
    private final int targetPiece;

    // Center piece is at row=8, col=7 in a 16x16 grid → pos = 8*16+7 = 135
    private static final int CENTER_POS = 135;

    // --- GPU VARIABLES ---
    private final List<int[]> gpuSeedBoards = new ArrayList<>();
    private final int HANDOFF_DEPTH = 50;
    private int forceBacktrackTarget = -1;

    public MasterSolverPBP(PieceInventory inventory, int trueCenterPiece) {
        this.inventory = inventory;
        this.targetPiece = trueCenterPiece;

        Arrays.fill(flatBoard, -1);
        Arrays.fill(bestBoard, -1);
        Arrays.fill(flatResumeBoard, -1);

        // Find physical index of center piece
        for (int i = 0; i < 1024; i++) {
            if (inventory.allOrientations[i] == targetPiece) {
                this.centerPhysicalIdx = inventory.physicalMapping[i];
                break;
            }
        }

        if (centerPhysicalIdx == -1) {
            System.err.println("WARNING: Center piece not found in inventory! Check CENTER_PIECE_INDEX and rotation.");
        } else {
            System.out.println("Center piece found at physical index " + centerPhysicalIdx
                + " | N=" + PieceUtils.getNorth(targetPiece)
                + " E=" + PieceUtils.getEast(targetPiece)
                + " S=" + PieceUtils.getSouth(targetPiece)
                + " W=" + PieceUtils.getWest(targetPiece));
        }

        int[][] loaded = CheckpointManager.load();
        if (loaded != null) {
            int validPieceCount = 0;
            int highestPosLoaded = -1;

            for (int i = 0; i < 256; i++) {
                int r = i / 16;
                int c = i % 16;
                if (i == CENTER_POS) continue; // center is always fixed, don't load from checkpoint

                if (loaded[r] != null) {
                    int p = loaded[r][c];
                    if (p != -1 && isValidPiece(p)) {
                        flatResumeBoard[i] = p;
                        bestBoard[i] = p;
                        validPieceCount++;
                        highestPosLoaded = Math.max(highestPosLoaded, i);
                    }
                }
            }
            if (validPieceCount > 0) {
                this.deepestPos = highestPosLoaded;
                bestBoard[CENTER_POS] = targetPiece;
                System.out.println(">>> PBP Checkpoint Loaded! Resuming from piece " + highestPosLoaded);
            }
        }
    }

    private boolean isValidPiece(int p) {
        if (p == -1) return false;
        for (int i = 0; i < 1024; i++) {
            if (inventory.allOrientations[i] == p) return true;
        }
        return false;
    }

    @Override
    public void run() {
        System.out.println("Starting Piece-By-Piece (PBP) Solver...");
        System.out.println("Center piece locked at position " + CENTER_POS + " (row=" + (CENTER_POS/16) + ", col=" + (CENTER_POS%16) + ")");

        // Lock the center piece before any search
        flatBoard[CENTER_POS] = targetPiece;
        if (centerPhysicalIdx != -1) usedPhysicalPieces[centerPhysicalIdx] = true;

        if (deepestPos > 0) updateDisplay(buildDisplayBoard(bestBoard));

        if (solve(0)) {
            System.out.println("SOLVED!");
        } else {
            System.out.println("Exhausted search space. Best: " + deepestPos + "/256 pieces.");
        }
    }

    private boolean solve(int pos) {
        // Skip the pre-placed center piece
        if (pos == CENTER_POS) return solve(pos + 1);

        if (pos == 256) return true;

        // GPU handoff at depth threshold
        if (pos == HANDOFF_DEPTH) {
            int[] clonedBoard = Arrays.copyOf(flatBoard, 256);
            gpuSeedBoards.add(clonedBoard);

            if (gpuSeedBoards.size() >= 10000) {
                runGpuHandoff(gpuSeedBoards, HANDOFF_DEPTH);
                gpuSeedBoards.clear();
                forceBacktrackTarget = 10 + rnd.nextInt(11);
            }
            return false;
        }

        // Track high score
        if (pos > deepestPos) {
            deepestPos = pos;
            System.arraycopy(flatBoard, 0, bestBoard, 0, 256);
            int[][] displayBoard = buildDisplayBoard(bestBoard);
            updateDisplay(displayBoard);
            RecordManager.saveRecord(displayBoard, deepestPos);
            CheckpointManager.save(displayBoard);
        }

        int row = pos / 16;
        int col = pos % 16;

        int n_req, w_req, s_req, e_req;

        // North: piece above is always placed (or board border)
        if (row == 0) {
            n_req = PieceUtils.BORDER_COLOR; // top border must be grey (0)
        } else {
            int above = flatBoard[pos - 16];
            n_req = (above != -1) ? PieceUtils.getSouth(above) : PieceUtils.WILDCARD;
        }

        // West: piece to the left is always placed (or board border)
        if (col == 0) {
            w_req = PieceUtils.BORDER_COLOR; // left border must be grey (0)
        } else {
            int left = flatBoard[pos - 1];
            w_req = (left != -1) ? PieceUtils.getEast(left) : PieceUtils.WILDCARD;
        }

        // South: piece below is NOT yet placed — only constrain if on bottom border
        s_req = (row == 15) ? PieceUtils.BORDER_COLOR : PieceUtils.WILDCARD;

        // East: piece to the right is NOT yet placed — only constrain if on right border
        e_req = (col == 15) ? PieceUtils.BORDER_COLOR : PieceUtils.WILDCARD;

        // Select correct piece pool based on board position
        int borderCount = 0;
        if (row == 0 || row == 15) borderCount++;
        if (col == 0 || col == 15) borderCount++;

        List<Integer> pool;
        if (borderCount == 2) pool = inventory.corners;
        else if (borderCount == 1) pool = inventory.edges;
        else pool = inventory.interior;

        if (pool.isEmpty()) return false;

        int size = pool.size();
        int offset = rnd.nextInt(size);

        for (int i = 0; i < size; i++) {
            int orientationIdx = pool.get((i + offset) % size);
            int p = inventory.allOrientations[orientationIdx];
            int physicalIdx = inventory.physicalMapping[orientationIdx];

            // Checkpoint fast-forward: if we have a resume piece for this position, only try that piece
            if (flatResumeBoard[pos] != -1) {
                if (p != flatResumeBoard[pos]) continue;
            }

            if (usedPhysicalPieces[physicalIdx]) continue;

            if (matches(p, n_req, e_req, s_req, w_req)) {
                flatBoard[pos] = p;
                usedPhysicalPieces[physicalIdx] = true;
                flatResumeBoard[pos] = -1; // consumed this resume hint

                if (solve(pos + 1)) return true;

                flatBoard[pos] = -1;
                usedPhysicalPieces[physicalIdx] = false;

                // Branch jumping after GPU handoff
                if (forceBacktrackTarget != -1) {
                    if (pos > forceBacktrackTarget) {
                        return false;
                    } else if (pos == forceBacktrackTarget) {
                        forceBacktrackTarget = -1;
                        System.out.println(">>> Branch Jumper reset at piece " + pos + ". Exploring new path...");
                    }
                }
            }
        }

        return false;
    }

    private boolean matches(int p, int n, int e, int s, int w) {
        if (n != PieceUtils.WILDCARD && PieceUtils.getNorth(p) != n) return false;
        if (e != PieceUtils.WILDCARD && PieceUtils.getEast(p) != e) return false;
        if (s != PieceUtils.WILDCARD && PieceUtils.getSouth(p) != s) return false;
        return w == PieceUtils.WILDCARD || PieceUtils.getWest(p) == w;
    }

    private int[][] buildDisplayBoard(int[] sourceArray) {
        int[][] displayBoard = new int[16][16];
        for (int i = 0; i < 16; i++) Arrays.fill(displayBoard[i], -1);
        for (int i = 0; i < 256; i++) {
            if (sourceArray[i] == -1) continue;
            displayBoard[i / 16][i % 16] = sourceArray[i];
        }
        return displayBoard;
    }

    private void updateDisplay(int[][] displayBoard) {
        Main.updateDisplay(deepestPos, displayBoard);
    }

    // --- GPU HANDOFF (unchanged from original) ---
    private void runGpuHandoff(List<int[]> partialBoardsList, int startingPos) {
        System.out.println(">>> INITIATING GPU HANDOFF...");
        System.out.println("Shipping " + partialBoardsList.size() + " partial boards to VRAM...");

        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        CUmodule module = new CUmodule();
        cuModuleLoad(module, "SolveEternityKernel.ptx");
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "solvePBP");

        int numBoards = partialBoardsList.size();
        int[] flatBoards = new int[numBoards * 256];
        for (int i = 0; i < numBoards; i++) {
            System.arraycopy(partialBoardsList.get(i), 0, flatBoards, i * 256, 256);
        }

        CUdeviceptr d_partialBoards = new CUdeviceptr();
        cuMemAlloc(d_partialBoards, (long) numBoards * 256 * Sizeof.INT);
        cuMemcpyHtoD(d_partialBoards, Pointer.to(flatBoards), (long) numBoards * 256 * Sizeof.INT);

        CUdeviceptr d_allOrientations = new CUdeviceptr();
        cuMemAlloc(d_allOrientations, 1024L * Sizeof.INT);
        cuMemcpyHtoD(d_allOrientations, Pointer.to(inventory.allOrientations), 1024L * Sizeof.INT);

        CUdeviceptr d_physicalMapping = new CUdeviceptr();
        cuMemAlloc(d_physicalMapping, 1024L * Sizeof.INT);
        cuMemcpyHtoD(d_physicalMapping, Pointer.to(inventory.physicalMapping), 1024L * Sizeof.INT);

        CUdeviceptr d_solution = new CUdeviceptr();
        cuMemAlloc(d_solution, 256L * Sizeof.INT);

        int[] solvedFlag = {0};
        CUdeviceptr d_solvedFlag = new CUdeviceptr();
        cuMemAlloc(d_solvedFlag, Sizeof.INT);
        cuMemcpyHtoD(d_solvedFlag, Pointer.to(solvedFlag), Sizeof.INT);

        int[] gpuScore = {0};
        CUdeviceptr d_gpuHighScore = new CUdeviceptr();
        cuMemAlloc(d_gpuHighScore, Sizeof.INT);
        cuMemcpyHtoD(d_gpuHighScore, Pointer.to(gpuScore), Sizeof.INT);

        CUdeviceptr d_bestBoardOut = new CUdeviceptr();
        cuMemAlloc(d_bestBoardOut, 256L * Sizeof.INT);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(d_partialBoards),
                Pointer.to(new int[]{numBoards}),
                Pointer.to(new int[]{startingPos}),
                Pointer.to(d_allOrientations),
                Pointer.to(d_physicalMapping),
                Pointer.to(d_solution),
                Pointer.to(d_solvedFlag),
                Pointer.to(d_gpuHighScore),
                Pointer.to(d_bestBoardOut)
        );

        int threadsPerBlock = 256;
        int blocksPerGrid = (int) Math.ceil((double) numBoards / threadsPerBlock);

        System.out.println("LAUNCHING CUDA KERNEL: " + blocksPerGrid + " blocks x " + threadsPerBlock + " threads.");
        long startTime = System.currentTimeMillis();

        cuLaunchKernel(function,
                blocksPerGrid, 1, 1,
                threadsPerBlock, 1, 1,
                0, null,
                kernelParameters, null);
        cuCtxSynchronize();

        long timeTaken = System.currentTimeMillis() - startTime;

        cuMemcpyDtoH(Pointer.to(solvedFlag), d_solvedFlag, Sizeof.INT);
        if (solvedFlag[0] == 1) {
            System.out.println(">>> GPU FOUND THE SOLUTION IN " + timeTaken + "ms! <<<");
            int[] winningBoard = new int[256];
            cuMemcpyDtoH(Pointer.to(winningBoard), d_solution, 256L * Sizeof.INT);
            updateDisplay(buildDisplayBoard(winningBoard));
            RecordManager.saveRecord(buildDisplayBoard(winningBoard), 256);
            System.exit(0);
        }

        cuMemcpyDtoH(Pointer.to(gpuScore), d_gpuHighScore, Sizeof.INT);
        System.out.println("GPU Batch done in " + timeTaken + "ms. Deepest GPU dive: " + gpuScore[0] + " pieces.");

        if (gpuScore[0] > deepestPos) {
            deepestPos = gpuScore[0];
            int[] gpuWinningBoard = new int[256];
            cuMemcpyDtoH(Pointer.to(gpuWinningBoard), d_bestBoardOut, 256L * Sizeof.INT);
            System.arraycopy(gpuWinningBoard, 0, bestBoard, 0, 256);
            int[][] displayBoard = buildDisplayBoard(bestBoard);
            updateDisplay(displayBoard);
            RecordManager.saveRecord(displayBoard, deepestPos);
            CheckpointManager.save(displayBoard);
            System.out.println(">>> GPU broke the CPU record! Saved checkpoint.");
        }

        cuMemFree(d_partialBoards);
        cuMemFree(d_allOrientations);
        cuMemFree(d_physicalMapping);
        cuMemFree(d_solution);
        cuMemFree(d_solvedFlag);
        cuMemFree(d_gpuHighScore);
        cuMemFree(d_bestBoardOut);
    }
}