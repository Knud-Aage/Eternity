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

    // --- GPU VARIABLES ---
    private final List<int[]> gpuSeedBoards = new ArrayList<>();
    private final int HANDOFF_DEPTH = 50;
    private int forceBacktrackTarget = -1;

    public MasterSolverPBP(PieceInventory inventory) {
        this.inventory = inventory;

        Arrays.fill(flatBoard, -1);
        Arrays.fill(bestBoard, -1);
        Arrays.fill(flatResumeBoard, -1);

        int targetPiece = PieceUtils.pack(18, 12, 18, 3);
        for (int i = 0; i < 1024; i++) {
            if (inventory.allOrientations[i] == targetPiece) {
                this.centerPhysicalIdx = inventory.physicalMapping[i];
                break;
            }
        }

        int[][] loaded = CheckpointManager.load();
        if (loaded != null) {
            int validPieceCount = 0;
            int highestPosLoaded = -1;

            for (int i = 0; i < 256; i++) {
                int gRow = i / 16;
                int gCol = i % 16;
                int mIdx = (gRow / 4) * 4 + (gCol / 4);
                int pIdx = (gRow % 4) * 4 + (gCol % 4);

                if (loaded[mIdx] != null) {
                    int p = loaded[mIdx][pIdx];
                    if (isValidPiece(p) && p != targetPiece) {
                        flatResumeBoard[i] = p;
                        bestBoard[i] = p;
                        validPieceCount++;
                        highestPosLoaded = Math.max(highestPosLoaded, i);
                    }
                }
            }
            if (validPieceCount > 0) {
                this.deepestPos = highestPosLoaded;
                bestBoard[119] = targetPiece;
                System.out.println(">>> PBP Checkpoint Loaded! Fast-forwarding up to piece " + highestPosLoaded + "...");
            }
        }
    }

    private boolean isValidPiece(int p) {
        for(int i = 0; i < 1024; i++) {
            if(inventory.allOrientations[i] == p) return true;
        }
        return false;
    }

    // --- THE JCUDA BRIDGE ---
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

        int[] solvedFlag = { 0 };
        CUdeviceptr d_solvedFlag = new CUdeviceptr();
        cuMemAlloc(d_solvedFlag, Sizeof.INT);
        cuMemcpyHtoD(d_solvedFlag, Pointer.to(solvedFlag), Sizeof.INT);

        int[] gpuScore = { 0 };
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

        System.out.println("LAUNCHING CUDA KERNEL: " + blocksPerGrid + " Blocks, " + threadsPerBlock + " Threads.");
        long startTime = System.currentTimeMillis();

        cuLaunchKernel(function,
                blocksPerGrid, 1, 1,
                threadsPerBlock, 1, 1,
                0, null,
                kernelParameters, null
        );
        cuCtxSynchronize();

        long timeTaken = System.currentTimeMillis() - startTime;

        cuMemcpyDtoH(Pointer.to(solvedFlag), d_solvedFlag, Sizeof.INT);
        if (solvedFlag[0] == 1) {
            System.out.println(">>> GPU FOUND THE SOLUTION IN " + timeTaken + "ms! <<<");
            int[] winningBoard = new int[256];
            cuMemcpyDtoH(Pointer.to(winningBoard), d_solution, 256L * Sizeof.INT);
            updateDisplay(buildLegacyBoard(winningBoard));
            RecordManager.saveRecord(buildLegacyBoard(winningBoard), 256);
            System.exit(0);
        }

        cuMemcpyDtoH(Pointer.to(gpuScore), d_gpuHighScore, Sizeof.INT);
        System.out.println("GPU Batch Finished in " + timeTaken + "ms! Deepest dive: " + gpuScore[0] + " pieces.");

        if (gpuScore[0] > deepestPos) {
            deepestPos = gpuScore[0];
            int[] gpuWinningBoard = new int[256];
            cuMemcpyDtoH(Pointer.to(gpuWinningBoard), d_bestBoardOut, 256L * Sizeof.INT);
            System.arraycopy(gpuWinningBoard, 0, bestBoard, 0, 256);

            int[][] legacyBoard = buildLegacyBoard(bestBoard);
            updateDisplay(legacyBoard);
            RecordManager.saveRecord(legacyBoard, deepestPos);
            CheckpointManager.save(legacyBoard);

            System.out.println(">>> GPU BROKE THE RECORD! Image and Checkpoint permanently saved.");
        }

        cuMemFree(d_partialBoards);
        cuMemFree(d_allOrientations);
        cuMemFree(d_physicalMapping);
        cuMemFree(d_solution);
        cuMemFree(d_solvedFlag);
        cuMemFree(d_gpuHighScore);
        cuMemFree(d_bestBoardOut);
    }

    @Override
    public void run() {
        System.out.println("Starting Piece-By-Piece (PBP) Solver...");

        flatBoard[119] = PieceUtils.pack(18, 12, 18, 3);
        if (centerPhysicalIdx != -1) usedPhysicalPieces[centerPhysicalIdx] = true;

        if (deepestPos > 0) updateDisplay(buildLegacyBoard(bestBoard));

        if (solve(0)) System.out.println("SOLVED!");
        else System.out.println("Exhausted search space. No solution found.");
    }

    private boolean solve(int pos) {
        if (pos == 119) return solve(pos + 1);

        // --- GPU HANDOFF TRIGGER ---
        if (pos == HANDOFF_DEPTH) {
            int[] clonedBoard = new int[256];
            System.arraycopy(flatBoard, 0, clonedBoard, 0, 256);
            gpuSeedBoards.add(clonedBoard);

            // Send the cargo ship as soon as we have 10,000!
            if (gpuSeedBoards.size() >= 10000) {
                runGpuHandoff(gpuSeedBoards, HANDOFF_DEPTH);
                gpuSeedBoards.clear();

                // We just shipped 10,000 seeds. Assume this timeline is dead!
                // Force the CPU to violently rewind to a random piece between 10 and 20.
                forceBacktrackTarget = 10 + rnd.nextInt(11);
            }
            return false;
        }

        if (pos == 256) return true;

        // --- HIGH SCORE & INSTANT CHECKPOINT ---
        if (pos > deepestPos) {
            deepestPos = pos;
            System.arraycopy(flatBoard, 0, bestBoard, 0, 256);
            int[][] legacyBoard = buildLegacyBoard(bestBoard);
            updateDisplay(legacyBoard);
            RecordManager.saveRecord(legacyBoard, deepestPos);
            CheckpointManager.save(legacyBoard);
        }

        int row = pos / 16;
        int col = pos % 16;

        int n_req = (row == 0) ? 0 : (flatBoard[pos - 16] != -1 ? PieceUtils.getSouth(flatBoard[pos - 16]) : PieceUtils.WILDCARD);
        int s_req = (row == 15) ? 0 : (flatBoard[pos + 16] != -1 ? PieceUtils.getNorth(flatBoard[pos + 16]) : PieceUtils.WILDCARD);
        int w_req = (col == 0) ? 0 : (flatBoard[pos - 1] != -1 ? PieceUtils.getEast(flatBoard[pos - 1]) : PieceUtils.WILDCARD);
        int e_req = (col == 15) ? 0 : (flatBoard[pos + 1] != -1 ? PieceUtils.getWest(flatBoard[pos + 1]) : PieceUtils.WILDCARD);

        int b_req = 0;
        if (row == 0 || row == 15) b_req++;
        if (col == 0 || col == 15) b_req++;

        List<Integer> pool;
        if (b_req == 2) pool = inventory.corners;
        else if (b_req == 1) pool = inventory.edges;
        else pool = inventory.interior;

        if (pool.isEmpty()) return false;

        int size = pool.size();
        int offset = rnd.nextInt(size);

        for (int i = 0; i < size; i++) {
            int orientationIdx = pool.get((i + offset) % size);
            int p = inventory.allOrientations[orientationIdx];
            int physicalIdx = inventory.physicalMapping[orientationIdx];

            if (flatResumeBoard[pos] != -1) {
                if (p != flatResumeBoard[pos]) continue;
            }

            if (usedPhysicalPieces[physicalIdx]) continue;

            if (matches(p, n_req, e_req, s_req, w_req)) {

                flatBoard[pos] = p;
                usedPhysicalPieces[physicalIdx] = true;

                int ghostPiece = flatResumeBoard[pos];
                flatResumeBoard[pos] = -1;

                if (solve(pos + 1)) return true;

                // BACKTRACK!
                flatBoard[pos] = -1;
                usedPhysicalPieces[physicalIdx] = false;

                // <--- NEW: STACK COLLAPSE LOGIC --->
                if (forceBacktrackTarget != -1) {
                    if (pos > forceBacktrackTarget) {
                        // We haven't gone deep enough yet. Keep collapsing!
                        return false;
                    } else if (pos == forceBacktrackTarget) {
                        // We arrived at the target! Turn off the jumper and resume normal solving.
                        forceBacktrackTarget = -1;
                        System.out.println(">>> Branch Jumper wiped the board back to piece " + pos + ". Exploring new timeline...");
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

    private int[][] buildLegacyBoard(int[] sourceArray) {
        int[][] legacyBoard = new int[16][];
        for (int i = 0; i < 256; i++) {
            if (sourceArray[i] == -1) continue;

            int gRow = i / 16;
            int gCol = i % 16;

            int mIdx = (gRow / 4) * 4 + (gCol / 4);
            int pIdx = (gRow % 4) * 4 + (gCol % 4);

            if (legacyBoard[mIdx] == null) {
                legacyBoard[mIdx] = new int[16];
                Arrays.fill(legacyBoard[mIdx], PieceUtils.pack(PieceUtils.WILDCARD, PieceUtils.WILDCARD, PieceUtils.WILDCARD, PieceUtils.WILDCARD));
            }
            legacyBoard[mIdx][pIdx] = sourceArray[i];
        }
        return legacyBoard;
    }

    private void updateDisplay(int[][] legacyBoard) {
        Main.updateDisplay(deepestPos, legacyBoard);
    }
}