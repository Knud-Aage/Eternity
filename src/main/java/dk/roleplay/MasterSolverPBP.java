package dk.roleplay;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

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
    private final int targetPiece;
    private final boolean useGpu; // <--- NEW: Knows if we are on a GPU or CPU
    private final List<int[]> gpuSeedBoards = new ArrayList<>();
    private final int HANDOFF_DEPTH = 50;
    private int centerPhysicalIdx = -1;
    private int deepestPos = 0;
    private int forceBacktrackTarget = -1;

    public MasterSolverPBP(PieceInventory inventory, int trueCenterPiece, boolean useGpu) {
        this.inventory = inventory;
        this.targetPiece = trueCenterPiece;
        this.useGpu = useGpu;

        Arrays.fill(flatBoard, -1);
        Arrays.fill(bestBoard, -1);
        Arrays.fill(flatResumeBoard, -1);

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

            // <--- Cleaned up to use standard 16x16 logic --->
            for (int i = 0; i < 256; i++) {
                int r = i / 16;
                int c = i % 16;

                if (loaded[r] != null) {
                    int p = loaded[r][c];
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
                bestBoard[135] = targetPiece; // Strictly placed at 135
                System.out.println(">>> PBP Checkpoint Loaded! Fast-forwarding up to piece " + highestPosLoaded + ".." +
                        ".");
            }
        }
    }

    private boolean isValidPiece(int p) {
        for (int i = 0; i < 1024; i++) {
            if (inventory.allOrientations[i] == p) {
                return true;
            }
        }
        return false;
    }

    private void runGpuHandoff(List<int[]> partialBoardsList, int startingPos) {
        System.out.println(">>> INITIATING GPU HANDOFF...");
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
        for (int i = 0; i < numBoards; i++) System.arraycopy(partialBoardsList.get(i), 0, flatBoards, i * 256, 256);

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

        if (gpuScore[0] > deepestPos) {
            deepestPos = gpuScore[0];
            int[] gpuWinningBoard = new int[256];
            cuMemcpyDtoH(Pointer.to(gpuWinningBoard), d_bestBoardOut, 256L * Sizeof.INT);
            System.arraycopy(gpuWinningBoard, 0, bestBoard, 0, 256);
            int[][] displayBoard = buildDisplayBoard(bestBoard);
            updateDisplay(displayBoard);
            RecordManager.saveRecord(displayBoard, deepestPos);
            CheckpointManager.save(displayBoard);
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

        flatBoard[135] = targetPiece;
        if (centerPhysicalIdx != -1) {
            usedPhysicalPieces[centerPhysicalIdx] = true;
        }

        if (deepestPos > 0) {
            updateDisplay(buildDisplayBoard(bestBoard));
        }

        if (solve(0)) {
            System.out.println("SOLVED!");
        } else {
            System.out.println("Exhausted search space. No solution found.");
        }
    }

    private boolean solve(int pos) {
        if (pos == 135) {
            return solve(pos + 1);
        }

        // <--- CPU SAFETY SWITCH: Completely skip CUDA commands if running on CPU! --->
        if (useGpu && pos == HANDOFF_DEPTH) {
            int[] clonedBoard = new int[256];
            System.arraycopy(flatBoard, 0, clonedBoard, 0, 256);
            gpuSeedBoards.add(clonedBoard);

            if (gpuSeedBoards.size() >= 10000) {
                runGpuHandoff(gpuSeedBoards, HANDOFF_DEPTH);
                gpuSeedBoards.clear();
                forceBacktrackTarget = 10 + rnd.nextInt(11);
            }
            return false;
        }

        if (pos == 256) {
            return true;
        }

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

        int n_req = (row == 0) ? 0 : (flatBoard[pos - 16] != -1 ? PieceUtils.getSouth(flatBoard[pos - 16]) :
                                      PieceUtils.WILDCARD);
        int s_req = (row == 15) ? 0 : (flatBoard[pos + 16] != -1 ? PieceUtils.getNorth(flatBoard[pos + 16]) :
                                       PieceUtils.WILDCARD);
        int w_req = (col == 0) ? 0 : (flatBoard[pos - 1] != -1 ? PieceUtils.getEast(flatBoard[pos - 1]) :
                                      PieceUtils.WILDCARD);
        int e_req = (col == 15) ? 0 : (flatBoard[pos + 1] != -1 ? PieceUtils.getWest(flatBoard[pos + 1]) :
                                       PieceUtils.WILDCARD);

        int b_req = 0;
        if (row == 0 || row == 15) {
            b_req++;
        }
        if (col == 0 || col == 15) {
            b_req++;
        }

        List<Integer> pool;
        if (b_req == 2) {
            pool = inventory.corners;
        } else if (b_req == 1) {
            pool = inventory.edges;
        } else {
            pool = inventory.interior;
        }

        if (pool.isEmpty()) {
            return false;
        }

        int size = pool.size();
        int offset = rnd.nextInt(size);

        for (int i = 0; i < size; i++) {
            int orientationIdx = pool.get((i + offset) % size);
            int p = inventory.allOrientations[orientationIdx];
            int physicalIdx = inventory.physicalMapping[orientationIdx];

            if (flatResumeBoard[pos] != -1) {
                if (p != flatResumeBoard[pos]) {
                    continue;
                }
            }

            if (usedPhysicalPieces[physicalIdx]) {
                continue;
            }

            if (matches(p, n_req, e_req, s_req, w_req)) {

                flatBoard[pos] = p;
                usedPhysicalPieces[physicalIdx] = true;

                int ghostPiece = flatResumeBoard[pos];
                flatResumeBoard[pos] = -1;

                if (solve(pos + 1)) {
                    return true;
                }

                flatBoard[pos] = -1;
                usedPhysicalPieces[physicalIdx] = false;

                if (forceBacktrackTarget != -1) {
                    if (pos > forceBacktrackTarget) {
                        return false;
                    } else if (pos == forceBacktrackTarget) {
                        forceBacktrackTarget = -1;
                    }
                }
            }
        }
        return false;
    }

    private boolean matches(int p, int n, int e, int s, int w) {
        if (n != PieceUtils.WILDCARD && PieceUtils.getNorth(p) != n) {
            return false;
        }
        if (e != PieceUtils.WILDCARD && PieceUtils.getEast(p) != e) {
            return false;
        }
        if (s != PieceUtils.WILDCARD && PieceUtils.getSouth(p) != s) {
            return false;
        }
        return w == PieceUtils.WILDCARD || PieceUtils.getWest(p) == w;
    }

    // <--- Draws the board cleanly left-to-right, top-to-bottom! --->
    private int[][] buildDisplayBoard(int[] sourceArray) {
        int[][] displayBoard = new int[16][16];
        for (int i = 0; i < 16; i++) {
            Arrays.fill(displayBoard[i], -1);
        }
        for (int i = 0; i < 256; i++) {
            if (sourceArray[i] == -1) {
                continue;
            }
            int r = i / 16;
            int c = i % 16;
            displayBoard[r][c] = sourceArray[i];
        }
        return displayBoard;
    }

    private void updateDisplay(int[][] displayBoard) {
        Main.updateDisplay(deepestPos, displayBoard);
    }
}