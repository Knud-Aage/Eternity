package dk.roleplay;

import java.time.Instant;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import static jcuda.driver.JCudaDriver.*;
import java.util.ArrayList;

public class MasterSolverPBP implements Runnable {
    private final PieceInventory inventory;

    private final int[] flatBoard = new int[256];
    private final int[] bestBoard = new int[256];
    private final int[] flatResumeBoard = new int[256];
    private final boolean[] usedPhysicalPieces = new boolean[256];
    private final Random rnd = new Random();
    private int centerPhysicalIdx = -1;
    private int deepestPos = 0;
    private final List<int[]> gpuSeedBoards = new ArrayList<>();
    private final int HANDOFF_DEPTH = 80; // The piece where the CPU hands off to the GPU

    public MasterSolverPBP(PieceInventory inventory) {
        this.inventory = inventory;

        // --- THE CRITICAL FIX: FILL ALL ARRAYS WITH -1 ---
        // Without this, Java defaults them to 0, which instantly breaks the constraints!
        Arrays.fill(flatBoard, -1);
        Arrays.fill(bestBoard, -1);
        Arrays.fill(flatResumeBoard, -1);

        // Find the physical ID of the true Centerpiece (N=18, E=12, S=18, W=3)
        int targetPiece = PieceUtils.pack(18, 12, 18, 3);
        for (int i = 0; i < 1024; i++) {
            if (inventory.allOrientations[i] == targetPiece) {
                this.centerPhysicalIdx = inventory.physicalMapping[i];
                break;
            }
        }

        int[][] loaded = CheckpointManager.load();
        if (loaded != null) {
            int highestPosLoaded = -1;
            // Translate the 2D Macro-Tile array back into our 1D PBP array
            for (int i = 0; i < 256; i++) {
                int gRow = i / 16;
                int gCol = i % 16;
                int mIdx = (gRow / 4) * 4 + (gCol / 4);
                int pIdx = (gRow % 4) * 4 + (gCol % 4);

                if (loaded[mIdx] != null) {
                    int p = loaded[mIdx][pIdx];
                    // Verify it is a real piece and not a Wildcard filler
                    if (isValidPiece(p) && p != targetPiece) {
                        flatResumeBoard[i] = p;
                        bestBoard[i] = p; // Pre-fill the High Score memory
                        highestPosLoaded = Math.max(highestPosLoaded, i);
                    }
                }
            }
            if (highestPosLoaded > -1) {
                this.deepestPos = highestPosLoaded;
                bestBoard[119] = targetPiece; // Don't forget the center piece!
                System.out.println(">>> PBP Checkpoint Loaded! Fast-forwarding up to piece " + highestPosLoaded + "...");
            }
        }
    }

    // --- THE JCUDA BRIDGE ---
    private void runGpuHandoff(List<int[]> partialBoardsList, int startingPos) {
        System.out.println(">>> INITIATING GPU HANDOFF...");
        System.out.println("Shipping " + partialBoardsList.size() + " partial boards to VRAM...");

        // 1. Initialize JCuda
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // 2. Load the Compiled C++ Kernel
        CUmodule module = new CUmodule();
        cuModuleLoad(module, "SolveEternityKernel.ptx");
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "solvePBP");

        // 3. Flatten the Partial Boards for the GPU
        int numBoards = partialBoardsList.size();
        int[] flatBoards = new int[numBoards * 256];
        for (int i = 0; i < numBoards; i++) {
            System.arraycopy(partialBoardsList.get(i), 0, flatBoards, i * 256, 256);
        }

        // 4. Allocate GPU VRAM (Device Pointers)
        CUdeviceptr d_partialBoards = new CUdeviceptr();
        cuMemAlloc(d_partialBoards, (long) numBoards * 256 * Sizeof.INT);
        cuMemcpyHtoD(d_partialBoards, Pointer.to(flatBoards), (long) numBoards * 256 * Sizeof.INT);

        CUdeviceptr d_allOrientations = new CUdeviceptr();
        cuMemAlloc(d_allOrientations, 1024L * Sizeof.INT);
        cuMemcpyHtoD(d_allOrientations, Pointer.to(inventory.allOrientations), 1024L * Sizeof.INT);

        CUdeviceptr d_physicalMapping = new CUdeviceptr();
        cuMemAlloc(d_physicalMapping, 1024L * Sizeof.INT);
        cuMemcpyHtoD(d_physicalMapping, Pointer.to(inventory.physicalMapping), 1024L * Sizeof.INT);

        // Setup the WINNER Output Array
        CUdeviceptr d_solution = new CUdeviceptr();
        cuMemAlloc(d_solution, 256L * Sizeof.INT);

        // Setup the Global Kill Switch Flag (starts at 0)
        int[] solvedFlag = { 0 };
        CUdeviceptr d_solvedFlag = new CUdeviceptr();
        cuMemAlloc(d_solvedFlag, Sizeof.INT);
        cuMemcpyHtoD(d_solvedFlag, Pointer.to(solvedFlag), Sizeof.INT);

        // Setup the GPU High Score Tracker
        int[] gpuScore = { 0 };
        CUdeviceptr d_gpuHighScore = new CUdeviceptr();
        cuMemAlloc(d_gpuHighScore, Sizeof.INT);
        cuMemcpyHtoD(d_gpuHighScore, Pointer.to(gpuScore), Sizeof.INT);
        CUdeviceptr d_bestBoardOut = new CUdeviceptr();
        cuMemAlloc(d_bestBoardOut, 256L * Sizeof.INT);

        // 5. Package the Arguments for C++
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

        // 6. Calculate Grid/Block sizes (50,000 threads)
        int threadsPerBlock = 256;
        int blocksPerGrid = (int) Math.ceil((double) numBoards / threadsPerBlock);

        // 7. FIRE THE KERNEL!
        System.out.println(Instant.now() + ": LAUNCHING CUDA KERNEL: " + blocksPerGrid + " Blocks, " + threadsPerBlock + " Threads.");
        long startTime = System.currentTimeMillis();

        cuLaunchKernel(function,
                blocksPerGrid, 1, 1,      // Grid dimension
                threadsPerBlock, 1, 1,    // Block dimension
                0, null,                  // Shared memory size and stream
                kernelParameters, null    // Kernel parameters
        );
        cuCtxSynchronize(); // Wait for all 50,000 threads to finish!

        cuMemcpyDtoH(Pointer.to(gpuScore), d_gpuHighScore, Sizeof.INT);
        System.out.println(Instant.now() + ": GPU Batch Finished! Deepest dive this run: " + gpuScore[0] + " pieces.");

        if (gpuScore[0] > deepestPos) {
            deepestPos = gpuScore[0];

            // Pull the winning board out of VRAM
            int[] gpuWinningBoard = new int[256];
            cuMemcpyDtoH(Pointer.to(gpuWinningBoard), d_bestBoardOut, 256L * Sizeof.INT);

            // Overwrite Java's memory with the GPU's masterpiece
            System.arraycopy(gpuWinningBoard, 0, bestBoard, 0, 256);

            // Translate it, paint the screen, and permanently save the image and checkpoint!
            int[][] legacyBoard = buildLegacyBoard(bestBoard);
            updateDisplay(legacyBoard);
            RecordManager.saveRecord(legacyBoard, deepestPos);
            CheckpointManager.save(legacyBoard);

            System.out.println(">>> GPU BROKE THE RECORD! Image and Checkpoint permanently saved.");
        }
        long timeTaken = System.currentTimeMillis() - startTime;

        // 8. Check if the GPU won!
        cuMemcpyDtoH(Pointer.to(solvedFlag), d_solvedFlag, Sizeof.INT);
        if (solvedFlag[0] == 1) {
            System.out.println(Instant.now() + ": >>> GPU FOUND THE SOLUTION IN " + timeTaken + "ms! <<<");
            int[] winningBoard = new int[256];
            cuMemcpyDtoH(Pointer.to(winningBoard), d_solution, 256L * Sizeof.INT);

            // Save the ultimate record!
            updateDisplay(buildLegacyBoard(winningBoard));
            RecordManager.saveRecord(buildLegacyBoard(winningBoard), 256);
            System.exit(0);
        } else {
            System.out.println(Instant.now() + ": GPU exhausted all branches in " + timeTaken + "ms. No solution found on this run.");
        }

        // 9. Free VRAM to prevent memory leaks
        cuMemFree(d_partialBoards);
        cuMemFree(d_allOrientations);
        cuMemFree(d_physicalMapping);
        cuMemFree(d_solution);
        cuMemFree(d_solvedFlag);
        cuMemFree(d_gpuHighScore);
        cuMemFree(d_bestBoardOut);
    }

    private boolean isValidPiece(int p) {
        for(int i = 0; i < 1024; i++) {
            if(inventory.allOrientations[i] == p) return true;
        }
        return false;
    }

    @Override
    public void run() {
        System.out.println("Starting Piece-By-Piece (PBP) Solver...");

        flatBoard[119] = PieceUtils.pack(18, 12, 18, 3);
        if (centerPhysicalIdx != -1) {
            usedPhysicalPieces[centerPhysicalIdx] = true;
        }

        // Force the GUI to show the loaded checkpoint immediately
        if (deepestPos > 0) {
            updateDisplay(buildLegacyBoard(bestBoard));
        }

        // Start the solver at the top-left corner (position 0)
        if (solve(0)) {
            System.out.println("SOLVED!");
        } else {
            System.out.println("Exhausted search space. No solution found.");
        }
    }

    private boolean solve(int pos) {
        if (pos == 119) return solve(pos + 1);

        // --- GPU HANDOFF TRIGGER ---
        if (pos == HANDOFF_DEPTH) {
            // We reached depth 80! Save this board.
            int[] clonedBoard = new int[256];
            System.arraycopy(flatBoard, 0, clonedBoard, 0, 256);
            gpuSeedBoards.add(clonedBoard);

            // Once we have 50,000 starting points, STOP THE CPU and fire the GPU!
            if (gpuSeedBoards.size() >= 50000) {
                runGpuHandoff(gpuSeedBoards, HANDOFF_DEPTH);
                gpuSeedBoards.clear(); // Clear memory after GPU finishes
            }

            // Force the CPU to backtrack and find a different path!
            return false;
        }

        // --- HIGH SCORE & RECORDS ---
        if (pos > deepestPos) {
            deepestPos = pos;

            // 1. Lock the board into memory
            System.arraycopy(flatBoard, 0, bestBoard, 0, 256);

            // 2. Translate it for the Visualizer
            int[][] legacyBoard = buildLegacyBoard(bestBoard);
            updateDisplay(legacyBoard);

            // 3. Save the image to the Records folder
            RecordManager.saveRecord(legacyBoard, deepestPos);

            // 4. INSTANTLY save the Checkpoint! (Timer removed)
            CheckpointManager.save(legacyBoard);
            System.out.println(">>> PBP Checkpoint Permanently Locked at: " + deepestPos + " pieces");
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

            // --- THE FAST-FORWARD FIX ---
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

                flatBoard[pos] = -1;
                usedPhysicalPieces[physicalIdx] = false;
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