package dk.roleplay;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import static jcuda.driver.JCudaDriver.*;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class MasterSolverPBP implements Runnable {

    public enum BuildStrategy {
        TYPEWRITER, SPIRAL
    }

    private static class EvolutionLeapException extends RuntimeException {}
    private static class PoisonedBaseCampException extends RuntimeException {}
    private static class ManualOverrideException extends RuntimeException {} // <--- NYT

    // GLOBALE CUDA VARIABLER
    private CUcontext cuContext;
    private CUmodule cuModule;
    private CUfunction cuFunction;

    private final PieceInventory inventory;
    private final int[] flatBoard = new int[256];
    private final int[] bestBoard = new int[256];
    private final int[] flatResumeBoard = new int[256];
    private final boolean[] usedPhysicalPieces = new boolean[256];
    private final Random rnd = new Random();
    private int centerPhysicalIdx = -1;

    private int deepestStep = 0;
    private int absoluteHighScore = 0;
    private final int targetPiece;

    private final boolean useGpu;
    private final BuildStrategy currentStrategy;
    private final boolean lockCenter;
    private final String saveProfile; // <--- NYT: Holder styr på de 4 profiler

    private final List<int[]> gpuSeedBoards = new ArrayList<>();

    private int HANDOFF_DEPTH;
    private int forceBacktrackTarget = -1;
    private int targetBatchSize = 10000;

    // <--- NYT: GUI KONTROL VARIABLER --->
    private volatile double extinctionThreshold = 0.98;
    private volatile boolean manualOverrideRequested = false;
    private volatile int manualBaseCampTarget = 0;

    private final int[] BUILD_ORDER;

    public MasterSolverPBP(PieceInventory inventory, int trueCenterPiece, boolean useGpu, BuildStrategy strategy, boolean lockCenter) {
        Arrays.fill(flatBoard, -1);
        Arrays.fill(bestBoard, -1);
        Arrays.fill(flatResumeBoard, -1);

        this.inventory = inventory;
        this.targetPiece = trueCenterPiece;
        this.useGpu = useGpu;
        this.currentStrategy = strategy;
        this.lockCenter = lockCenter;

        // Sørger for at Cheat og Non-Cheat filer ikke overskriver hinanden
        this.saveProfile = strategy.name() + (lockCenter ? "_LOCKED" : "_UNLOCKED");

        for (int i = 0; i < 1024; i++) {
            if (inventory.allOrientations[i] == targetPiece) {
                this.centerPhysicalIdx = inventory.physicalMapping[i];
                break;
            }
        }

        if (strategy == BuildStrategy.SPIRAL) {
            BUILD_ORDER = generateSpiralOrder();
            HANDOFF_DEPTH = 70;
        } else {
            BUILD_ORDER = generateTypewriterOrder();
            HANDOFF_DEPTH = 50;
        }

        int[][] loaded = CheckpointManager.load(saveProfile);
        if (loaded != null) {
            int highestStepLoaded = -1;
            for (int step = 0; step < 256; step++) {
                int boardIdx = BUILD_ORDER[step];
                int r = boardIdx / 16;
                int c = boardIdx % 16;

                if (loaded[r] != null) {
                    int p = loaded[r][c];
                    if (isValidPiece(p) && p != targetPiece) {
                        bestBoard[boardIdx] = p;
                        highestStepLoaded = Math.max(highestStepLoaded, step);
                    }
                }
            }
            if (highestStepLoaded > 0) {
                this.deepestStep = highestStepLoaded;
                this.absoluteHighScore = highestStepLoaded;
                if (lockCenter) bestBoard[135] = targetPiece;
            }
        }
    }

    // <--- NYT: METODER SOM GUI'EN KAN KALDE --->
    public void setExtinctionThreshold(double threshold) {
        this.extinctionThreshold = threshold;
    }

    public void triggerManualOverride(int targetBaseCamp) {
        this.manualBaseCampTarget = targetBaseCamp;
        this.manualOverrideRequested = true;
    }

    private void initCUDA() {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);

        cuContext = new CUcontext();
        cuCtxCreate(cuContext, 0, device);

        cuModule = new CUmodule();
        cuModuleLoad(cuModule, "SolveEternityKernel.ptx");

        cuFunction = new CUfunction();
        cuModuleGetFunction(cuFunction, cuModule, "solvePBP");
        System.out.printf("%s: CUDA>>> CUDA Context & Kernel indlæst succesfuldt! %n", LocalDateTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss")));
    }

    private static int[] generateTypewriterOrder() {
        int[] order = new int[256];
        for (int i = 0; i < 256; i++) order[i] = i;
        return order;
    }

    private static int[] generateSpiralOrder() {
        int[] order = new int[256];
        boolean[][] visited = new boolean[16][16];
        int r = 0, c = 0;
        int[] dr = {0, 1, 0, -1};
        int[] dc = {1, 0, -1, 0};
        int dir = 0;
        for (int i = 0; i < 256; i++) {
            order[i] = r * 16 + c;
            visited[r][c] = true;
            int nr = r + dr[dir];
            int nc = c + dc[dir];
            if (nr < 0 || nr >= 16 || nc < 0 || nc >= 16 || visited[nr][nc]) {
                dir = (dir + 1) % 4;
                nr = r + dr[dir];
                nc = c + dc[dir];
            }
            r = nr;
            c = nc;
        }
        return order;
    }

    private boolean isValidPiece(int p) {
        for(int i = 0; i < 1024; i++) {
            if(inventory.allOrientations[i] == p) return true;
        }
        return false;
    }

    private void runGpuHandoff(List<int[]> partialBoardsList, int startingStep) {
        int numBoards = partialBoardsList.size();
        if (numBoards == 0) return;

        int[] flatBoards = new int[numBoards * 256];
        for (int i = 0; i < numBoards; i++) System.arraycopy(partialBoardsList.get(i), 0, flatBoards, i * 256, 256);

        CUdeviceptr d_partialBoards = new CUdeviceptr();
        cuMemAlloc(d_partialBoards, (long) numBoards * 256 * Sizeof.INT);
        cuMemcpyHtoD(d_partialBoards, Pointer.to(flatBoards), (long) numBoards * 256 * Sizeof.INT);

        CUdeviceptr d_buildOrder = new CUdeviceptr();
        cuMemAlloc(d_buildOrder, 256L * Sizeof.INT);
        cuMemcpyHtoD(d_buildOrder, Pointer.to(BUILD_ORDER), 256L * Sizeof.INT);

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

        long[] totalSteps = { 0L };
        CUdeviceptr d_totalSteps = new CUdeviceptr();
        cuMemAlloc(d_totalSteps, 8L);
        cuMemcpyHtoD(d_totalSteps, Pointer.to(totalSteps), 8L);

        int[] threadDepths = new int[numBoards];
        CUdeviceptr d_threadDepths = new CUdeviceptr();
        cuMemAlloc(d_threadDepths, (long) numBoards * Sizeof.INT);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(d_partialBoards),
                Pointer.to(new int[]{numBoards}),
                Pointer.to(new int[]{startingStep}),
                Pointer.to(d_buildOrder),
                Pointer.to(d_allOrientations),
                Pointer.to(d_physicalMapping),
                Pointer.to(d_solution),
                Pointer.to(d_solvedFlag),
                Pointer.to(d_gpuHighScore),
                Pointer.to(d_bestBoardOut),
                Pointer.to(d_totalSteps),
                Pointer.to(new int[]{ lockCenter ? 1 : 0 }),
                Pointer.to(d_threadDepths)
        );

        int threadsPerBlock = 256;
        int blocksPerGrid = (int) Math.ceil((double) numBoards / threadsPerBlock);

        long startTime = System.currentTimeMillis();
        cuLaunchKernel(cuFunction, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, null, kernelParameters, null);
        cuCtxSynchronize();
        long timeTaken = System.currentTimeMillis() - startTime;

        cuMemcpyDtoH(Pointer.to(totalSteps), d_totalSteps, 8L);
        cuMemcpyDtoH(Pointer.to(threadDepths), d_threadDepths, (long) numBoards * Sizeof.INT);

        double timeSeconds = Math.max(timeTaken / 1000.0, 0.001);
        double speed = totalSteps[0] / timeSeconds;

        long sumDepth = 0;
        int maxInBatch = 0;
        int deadOnArrival = 0;
        for (int d : threadDepths) {
            sumDepth += d;
            if (d > maxInBatch) maxInBatch = d;
            if (d <= startingStep + 5) deadOnArrival++;
        }

        System.out.printf("%s: GPU Batch | %d ms | %,.0f pcs/s | Avg Depth: %.1f | Max: %d | Dead < 5: %d/%d\n",
                LocalDateTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss")), timeTaken, speed, (double)sumDepth/numBoards, maxInBatch, deadOnArrival, numBoards);

        if (numBoards > 0 && (double) deadOnArrival / numBoards >= extinctionThreshold) {
            cuFreeResources(d_partialBoards, d_buildOrder, d_allOrientations, d_physicalMapping, d_solution, d_solvedFlag, d_gpuHighScore, d_bestBoardOut, d_totalSteps, d_threadDepths);
            throw new PoisonedBaseCampException();
        }

        cuMemcpyDtoH(Pointer.to(solvedFlag), d_solvedFlag, Sizeof.INT);
        if (solvedFlag[0] == 1) {
            System.out.println(">>> GPU FOUND THE SOLUTION! <<<");
            int[] winningBoard = new int[256];
            cuMemcpyDtoH(Pointer.to(winningBoard), d_solution, 256L * Sizeof.INT);
            updateDisplay(buildDisplayBoard(winningBoard));
            RecordManager.saveRecord(buildDisplayBoard(winningBoard), 256, saveProfile);
            System.exit(0);
        }

        cuMemcpyDtoH(Pointer.to(gpuScore), d_gpuHighScore, Sizeof.INT);

        if (gpuScore[0] > deepestStep) {
            deepestStep = gpuScore[0];
            int[] gpuWinningBoard = new int[256];
            cuMemcpyDtoH(Pointer.to(gpuWinningBoard), d_bestBoardOut, 256L * Sizeof.INT);
            System.arraycopy(gpuWinningBoard, 0, bestBoard, 0, 256);

            int[][] displayBoard = buildDisplayBoard(bestBoard);
            updateDisplay(displayBoard);

            if (deepestStep > absoluteHighScore) {
                absoluteHighScore = deepestStep;
                RecordManager.saveRecord(displayBoard, absoluteHighScore, saveProfile);
                CheckpointManager.save(displayBoard, saveProfile);
            }

            if (currentStrategy == BuildStrategy.SPIRAL && deepestStep > HANDOFF_DEPTH + 30) {
                cuFreeResources(d_partialBoards, d_buildOrder, d_allOrientations, d_physicalMapping, d_solution, d_solvedFlag, d_gpuHighScore, d_bestBoardOut, d_totalSteps, d_threadDepths);
                throw new EvolutionLeapException();
            }
        }

        cuFreeResources(d_partialBoards, d_buildOrder, d_allOrientations, d_physicalMapping, d_solution, d_solvedFlag, d_gpuHighScore, d_bestBoardOut, d_totalSteps, d_threadDepths);
    }

    private void cuFreeResources(CUdeviceptr... pointers) {
        for (CUdeviceptr ptr : pointers) {
            cuMemFree(ptr);
        }
    }

    @Override
    public void run() {
        System.out.println("Starting Engine (Profile: " + saveProfile + ")...");

        if (useGpu) {
            initCUDA();
        }

        while (true) {
            Arrays.fill(flatBoard, -1);
            Arrays.fill(flatResumeBoard, -1);
            Arrays.fill(usedPhysicalPieces, false);
            gpuSeedBoards.clear();

            if (deepestStep > 0) updateDisplay(buildDisplayBoard(bestBoard));

            if (lockCenter) {
                flatBoard[135] = targetPiece;
                if (centerPhysicalIdx != -1) usedPhysicalPieces[centerPhysicalIdx] = true;
            }

            int lockedPieces = 0;
            if (deepestStep > 0 && currentStrategy == BuildStrategy.SPIRAL) {
                lockedPieces = Math.max(0, deepestStep - 30);

                int gap;
                if (lockedPieces > 180) {
                    gap = 2;
                    targetBatchSize = 50;
                } else if (lockedPieces > 150) {
                    gap = 4;
                    targetBatchSize = 250;
                } else {
                    gap = 15;
                    targetBatchSize = 10000;
                }

                HANDOFF_DEPTH = lockedPieces + gap;

                System.out.println("\n=======================================================");
                System.out.println(">>> EVOLUTION! Base Camp locked to step " + lockedPieces);
                System.out.println(">>> CPU building freely to Handoff at step " + HANDOFF_DEPTH);
                System.out.println(">>> Target seeds before GPU trigger: " + targetBatchSize);
                System.out.println("=======================================================\n");

                Arrays.fill(flatResumeBoard, -1);
                for (int step = 0; step < lockedPieces; step++) {
                    int boardIdx = BUILD_ORDER[step];
                    if (lockCenter && boardIdx == 135) continue;
                    flatResumeBoard[boardIdx] = bestBoard[boardIdx];
                }
            }

            boolean spaceExhausted = true;
            try {
                if (solve(0)) {
                    System.out.println("SOLVED!");
                    return;
                }
            } catch (EvolutionLeapException e) {
                spaceExhausted = false;
            } catch (PoisonedBaseCampException e) {
                spaceExhausted = false;
                if (lockedPieces > 70) {
                    deepestStep = Math.max(70, lockedPieces - 8);
                    System.out.println("\n[!] EXTINCTION EVENT! " + (extinctionThreshold*100) + "%+ seeds died instantly.");
                    System.out.println("[!] Poisoned Base Camp detected. Retreating to step " + deepestStep + "...\n");
                }
            } catch (ManualOverrideException e) {
                // <--- GUI OVERRIDE HÅNDTERING --->
                spaceExhausted = false;
                deepestStep = manualBaseCampTarget + 30; // Tvinger næste loop til at bruge dit tal!
                System.out.println("\n[!] MANUAL OVERRIDE! User forced base camp jump to step " + manualBaseCampTarget + ".\n");
            }

            if (spaceExhausted) {
                if (!gpuSeedBoards.isEmpty()) {
                    System.out.println(">>> Space exhausted early! Flushing remaining " + gpuSeedBoards.size() + " seeds to GPU...");
                    runGpuHandoff(gpuSeedBoards, HANDOFF_DEPTH);
                    gpuSeedBoards.clear();
                }

                if (currentStrategy == BuildStrategy.SPIRAL && lockedPieces > 0) {
                    deepestStep = Math.max(0, lockedPieces - 10);
                    System.out.println(">>> Base camp exhausted. Falling back to search new path...");
                } else {
                    System.out.println("Total search space exhausted. No solution found.");
                    return;
                }
            }
        }
    }

    private boolean solve(int step) {
        // <--- GUI OVERRIDE TJEK --->
        if (manualOverrideRequested) {
            manualOverrideRequested = false;
            throw new ManualOverrideException();
        }

        if (step == 256) return true;

        int boardIdx = BUILD_ORDER[step];

        if (lockCenter && boardIdx == 135) return solve(step + 1);

        if (useGpu && step == HANDOFF_DEPTH) {
            int[] clonedBoard = new int[256];
            System.arraycopy(flatBoard, 0, clonedBoard, 0, 256);
            gpuSeedBoards.add(clonedBoard);

            int printInterval = (targetBatchSize <= 250) ? 10 : 250;
//            if (gpuSeedBoards.size() % printInterval == 0) {
//                System.out.println("   [CPU Status] Fandt " + gpuSeedBoards.size() + " / " + targetBatchSize + " seeds...");
//            }

            if (gpuSeedBoards.size() >= targetBatchSize) {
                runGpuHandoff(gpuSeedBoards, HANDOFF_DEPTH);
                gpuSeedBoards.clear();

                if (currentStrategy == BuildStrategy.SPIRAL) {
                    forceBacktrackTarget = HANDOFF_DEPTH - 4;
                } else {
                    forceBacktrackTarget = 10 + rnd.nextInt(11);
                }
            }
            return false;
        }

        if (step > deepestStep) {
            deepestStep = step;
            System.arraycopy(flatBoard, 0, bestBoard, 0, 256);
            int[][] displayBoard = buildDisplayBoard(bestBoard);
            updateDisplay(displayBoard);

            if (deepestStep > absoluteHighScore) {
                absoluteHighScore = deepestStep;
                RecordManager.saveRecord(displayBoard, absoluteHighScore, saveProfile);
                CheckpointManager.save(displayBoard, saveProfile);
            }
        }

        int row = boardIdx / 16;
        int col = boardIdx % 16;

        int n_req = (row == 0) ? 0 : (flatBoard[boardIdx - 16] != -1 ? PieceUtils.getSouth(flatBoard[boardIdx - 16]) : PieceUtils.WILDCARD);
        int s_req = (row == 15) ? 0 : (flatBoard[boardIdx + 16] != -1 ? PieceUtils.getNorth(flatBoard[boardIdx + 16]) : PieceUtils.WILDCARD);
        int w_req = (col == 0) ? 0 : (flatBoard[boardIdx - 1] != -1 ? PieceUtils.getEast(flatBoard[boardIdx - 1]) : PieceUtils.WILDCARD);
        int e_req = (col == 15) ? 0 : (flatBoard[boardIdx + 1] != -1 ? PieceUtils.getWest(flatBoard[boardIdx + 1]) : PieceUtils.WILDCARD);

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

            if (flatResumeBoard[boardIdx] != -1) {
                if (p != flatResumeBoard[boardIdx]) continue;
            }

            if (usedPhysicalPieces[physicalIdx]) continue;

            if (matches(p, n_req, e_req, s_req, w_req)) {

                flatBoard[boardIdx] = p;
                usedPhysicalPieces[physicalIdx] = true;

                int ghostPiece = flatResumeBoard[boardIdx];
                flatResumeBoard[boardIdx] = -1;

                if (solve(step + 1)) return true;

                flatBoard[boardIdx] = -1;
                usedPhysicalPieces[physicalIdx] = false;

                if (forceBacktrackTarget != -1) {
                    if (step > forceBacktrackTarget) {
                        return false;
                    } else if (step == forceBacktrackTarget) {
                        forceBacktrackTarget = -1;
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
            int r = i / 16;
            int c = i % 16;
            displayBoard[r][c] = sourceArray[i];
        }
        return displayBoard;
    }

    private void updateDisplay(int[][] displayBoard) {
        Main.updateDisplay(absoluteHighScore, displayBoard);
    }
}