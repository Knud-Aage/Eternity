package dk.puzzle.gpu;

import dk.puzzle.blackwood.BwGpuTables;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.JCuda;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.*;

/**
 * GPU engine for the genuinely faithful Blackwood-native kernel ({@code SolveBlackwoodKernel.cu}
 * / {@code solveBlackwoodDfs}) -- deliberately separate from {@link GpuEngine}, which drives the
 * existing generic-index {@code solvePBP}/{@code solveRepairMode} kernels for the live production
 * pipeline. That class's Blackwood-bias machinery (candidate-order jitter tables biasing a
 * different, generic index) has no role here; mixing this engine's own memory layout and launch
 * cadence into it would add risk to the live pipeline for no reuse benefit.
 *
 * <p>Unlike {@link GpuEngine} (which uploads its constant tables once, in the constructor, since
 * they never change for the life of the engine), this engine's candidate tables are re-uploaded
 * via {@link #uploadTables} once per BATCH (= once per {@link #runBlackwoodDfs} call from the
 * caller's perspective) -- Blackwood's own algorithm re-randomizes the 10 batch-level tables'
 * score+jitter sort every outer {@code prepare()} pass, and this mirrors that cadence.</p>
 */
public class BlackwoodGpuEngine {

    private static final int MAX_THREADS = 20_000;
    // Headroom over the real, measured sizes (BwGpuTablesTest: payload ~38,675, bottomRawPayload
    // exactly 56) -- not the plan's rough pre-implementation estimate (~32,600). Keep these two
    // constants in sync with BwGpuTablesTest's own thresholds if either ever needs to move.
    private static final int MAX_PAYLOAD_SIZE = 50_000;
    private static final int MAX_BOTTOM_PAYLOAD_SIZE = 96;

    private CUfunction blackwoodDfsFunction;
    private CUmodule cuModule;

    // Persistent device buffers -- allocated once, reused every launch.
    private CUdeviceptr d_payload;
    private CUdeviceptr d_gpuHighScore;
    private CUdeviceptr d_bestBoardOut;
    private CUdeviceptr d_solution;
    private CUdeviceptr d_solvedFlag;
    private CUdeviceptr d_totalNodes;
    private CUdeviceptr d_threadDepths;

    public BlackwoodGpuEngine() {
        JCuda.cudaSetDeviceFlags(JCuda.cudaDeviceScheduleBlockingSync);
        initCUDA();
    }

    private void initCUDA() {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext cuContext = new CUcontext();
        cuCtxCreate(cuContext, 0, device);

        cuModule = new CUmodule();
        cuModuleLoad(cuModule, "SolveBlackwoodKernel.ptx");

        blackwoodDfsFunction = new CUfunction();
        cuModuleGetFunction(blackwoodDfsFunction, cuModule, "solveBlackwoodDfs");

        allocatePersistentBuffers();
    }

    private void allocatePersistentBuffers() {
        d_payload      = alloc((long) MAX_PAYLOAD_SIZE * Sizeof.INT);
        d_gpuHighScore = alloc(Sizeof.INT);
        d_bestBoardOut = alloc(256L * Sizeof.INT);
        d_solution     = alloc(256L * Sizeof.INT);
        d_solvedFlag   = alloc(Sizeof.INT);
        d_totalNodes   = alloc(Sizeof.LONG);
        d_threadDepths = alloc((long) MAX_THREADS * Sizeof.INT);
    }

    private static CUdeviceptr alloc(long bytes) {
        CUdeviceptr p = new CUdeviceptr();
        cuMemAlloc(p, bytes);
        return p;
    }

    private void uploadConstant(String symbol, int[] data, long bytes) {
        CUdeviceptr ptr = new CUdeviceptr();
        long[] size = new long[1];
        cuModuleGetGlobal(ptr, size, cuModule, symbol);
        cuMemcpyHtoD(ptr, Pointer.to(data), bytes);
    }

    /**
     * Uploads a fresh candidate-table set -- call once per batch, before {@link #runBlackwoodDfs}.
     * Unlike {@link GpuEngine}'s one-time constant uploads, this is meant to be called repeatedly
     * (once per {@code BlackwoodSolver.prepare()} + {@code BwGpuTables.build()} cycle).
     */
    public void uploadTables(BwGpuTables.GpuTableSet tables) {
        if (tables.payload().length > MAX_PAYLOAD_SIZE) {
            throw new IllegalStateException("candidate payload size " + tables.payload().length
                    + " exceeds MAX_PAYLOAD_SIZE=" + MAX_PAYLOAD_SIZE + " -- bump the buffer (see BwGpuTablesTest for the real measured size)");
        }
        if (tables.bottomRawPayload().length > MAX_BOTTOM_PAYLOAD_SIZE) {
            throw new IllegalStateException("bottomRawPayload size " + tables.bottomRawPayload().length
                    + " exceeds MAX_BOTTOM_PAYLOAD_SIZE=" + MAX_BOTTOM_PAYLOAD_SIZE + " -- bump the buffer");
        }

        long csrBytes = (long) BwGpuTables.NUM_TABLES * BwGpuTables.KEY_SPACE * Sizeof.INT;
        uploadConstant("c_csrOffset", tables.csrOffset(), csrBytes);
        uploadConstant("c_csrCount", tables.csrCount(), csrBytes);
        uploadConstant("c_bottomRawOffset", tables.bottomRawOffset(), 23L * Sizeof.INT);
        uploadConstant("c_bottomRawCount", tables.bottomRawCount(), 23L * Sizeof.INT);

        // c_bottomRawPayload is a fixed-size __constant__ array (MAX_BOTTOM_PAYLOAD=96 in the
        // .cu) -- pad the real, shorter payload with zeros rather than uploading a mismatched byte count.
        int[] paddedBottomPayload = Arrays.copyOf(tables.bottomRawPayload(), MAX_BOTTOM_PAYLOAD_SIZE);
        uploadConstant("c_bottomRawPayload", paddedBottomPayload, (long) MAX_BOTTOM_PAYLOAD_SIZE * Sizeof.INT);

        uploadConstant("c_stepToTableId", tables.stepToTableId(), 256L * Sizeof.INT);
        uploadConstant("c_stepBoardIdx", tables.stepBoardIdx(), 256L * Sizeof.INT);
        uploadConstant("c_breakArray", tables.breakArray(), 256L * Sizeof.INT);
        uploadConstant("c_heuristicArray", tables.heuristicArray(), 256L * Sizeof.INT);

        cuMemcpyHtoD(d_payload, Pointer.to(tables.payload()), (long) tables.payload().length * Sizeof.INT);
    }

    public record GpuResult(int newHighScore, boolean solved, long nodesTaken, int[] threadDepths) {
    }

    /**
     * Launches one batch: {@code numThreads} threads, each running one full Blackwood attempt
     * from a fresh step-0 seed (no CPU-supplied partial boards, unlike {@link GpuEngine#runDeepDfs}
     * -- Blackwood's algorithm always starts fresh). {@code bestBoardOut} is only overwritten if
     * this launch found a new high score or a genuine full solve -- same out-parameter convention
     * as {@code GpuEngine.runDeepDfs}, so check {@code newHighScore > currentHighScore} or
     * {@code solved} before trusting its contents.
     */
    public GpuResult runBlackwoodDfs(long seedBase, long stepBudget, int numThreads,
                                      int currentHighScore, int[] bestBoardOut) {
        if (numThreads > MAX_THREADS) {
            throw new IllegalArgumentException("numThreads " + numThreads + " exceeds MAX_THREADS=" + MAX_THREADS);
        }

        cuMemcpyHtoD(d_gpuHighScore, Pointer.to(new int[]{currentHighScore}), Sizeof.INT);
        cuMemcpyHtoD(d_solvedFlag, Pointer.to(new int[]{0}), Sizeof.INT);
        cuMemcpyHtoD(d_totalNodes, Pointer.to(new long[]{0L}), Sizeof.LONG);
        cuMemcpyHtoD(d_threadDepths, Pointer.to(new int[numThreads]), (long) numThreads * Sizeof.INT);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(d_payload),
                Pointer.to(new long[]{seedBase}),
                Pointer.to(new long[]{stepBudget}),
                Pointer.to(new int[]{numThreads}),
                Pointer.to(d_gpuHighScore),
                Pointer.to(d_bestBoardOut),
                Pointer.to(d_solution),
                Pointer.to(d_solvedFlag),
                Pointer.to(d_totalNodes),
                Pointer.to(d_threadDepths)
        );

        int blockSize = 256;
        int gridSize = (int) Math.ceil((double) numThreads / blockSize);
        cuLaunchKernel(blackwoodDfsFunction, gridSize, 1, 1, blockSize, 1, 1, 0, null, kernelParameters, null);
        cuCtxSynchronize();

        int[] resultHighScore = new int[1];
        long[] totalNodes = new long[1];
        int[] solved = new int[1];
        int[] threadDepths = new int[numThreads];

        cuMemcpyDtoH(Pointer.to(resultHighScore), d_gpuHighScore, Sizeof.INT);
        cuMemcpyDtoH(Pointer.to(totalNodes), d_totalNodes, Sizeof.LONG);
        cuMemcpyDtoH(Pointer.to(solved), d_solvedFlag, Sizeof.INT);
        cuMemcpyDtoH(Pointer.to(threadDepths), d_threadDepths, (long) numThreads * Sizeof.INT);

        if (resultHighScore[0] > currentHighScore) {
            cuMemcpyDtoH(Pointer.to(bestBoardOut), d_bestBoardOut, 256L * Sizeof.INT);
        }
        if (solved[0] == 1) {
            cuMemcpyDtoH(Pointer.to(bestBoardOut), d_solution, 256L * Sizeof.INT);
        }

        return new GpuResult(resultHighScore[0], solved[0] == 1, totalNodes[0], threadDepths);
    }
}
