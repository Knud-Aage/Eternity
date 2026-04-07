package dk.roleplay;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;
import jcuda.driver.JCudaDriver;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * GPU-accelerated implementation of CandidateValidator.
 * Utilizes JCuda to launch a CUDA kernel for high-throughput batch validation
 * of macro-tile candidates.
 */
public class GpuValidator implements CandidateValidator {
    private static final int MAX_RESULTS = 10000;
    private final CUfunction function;
    private final CUcontext context;

    /**
     * Initializes the CUDA driver, creates a device context, and loads the PTX module.
     *
     * @throws RuntimeException if CUDA initialization or module loading fails
     */
    public GpuValidator() {
        JCudaDriver.setExceptionsEnabled(false);
        int result = JCudaDriver.cuInit(0);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("cuInit failed: " + CUresult.stringFor(result));
        }

        CUdevice device = new CUdevice();
        JCudaDriver.cuDeviceGet(device, 0);
        this.context = new CUcontext();
        JCudaDriver.cuCtxCreate(this.context, 0, device);

        String ptxPath = "EternityKernel.ptx";
        if (!new File(ptxPath).exists()) {
            ptxPath = "src/main/resources/EternityKernel.ptx";
        }
        if (!new File(ptxPath).exists()) {
            ptxPath = "../EternityKernel.ptx";
        }

        CUmodule module = new CUmodule();
        result = JCudaDriver.cuModuleLoad(module, ptxPath);
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("cuModuleLoad failed with error code: " + CUresult.stringFor(result) +
                    " (" + result + "). Path: " + new File(ptxPath).getAbsolutePath() +
                    ". Make sure your GPU supports sm_75 and your driver is up to date (CUDA 13.2 requires Driver " +
                    "550+).");
        }

        function = new CUfunction();
        JCudaDriver.cuModuleGetFunction(function, module, "validateMacroTiles");
        JCudaDriver.setExceptionsEnabled(true);
    }

    /**
     * Transfers a candidate batch to the GPU, executes the validation kernel,
     * and retrieves the results.
     *
     * @param candidateBatch  Flattened 1D array of 32-bit packed integers
     * @param numPermutations Total number of 16-piece candidates in the batch
     * @return A list of validated 16-piece arrays that are internally consistent
     */
    @Override
    public List<int[]> validate(int[] candidateBatch, int numPermutations) {
        if (numPermutations == 0) {
            return new ArrayList<>();
        }

        JCudaDriver.cuCtxPushCurrent(context);

        try {
            CUdeviceptr d_candidates = new CUdeviceptr();
            JCudaDriver.cuMemAlloc(d_candidates, (long) candidateBatch.length * Sizeof.INT);
            JCudaDriver.cuMemcpyHtoD(d_candidates, Pointer.to(candidateBatch),
                    (long) candidateBatch.length * Sizeof.INT);

            CUdeviceptr d_results = new CUdeviceptr();
            JCudaDriver.cuMemAlloc(d_results, (long) MAX_RESULTS * 16 * Sizeof.INT);

            CUdeviceptr d_counter = new CUdeviceptr();
            JCudaDriver.cuMemAlloc(d_counter, Sizeof.INT);
            JCudaDriver.cuMemsetD32(d_counter, 0, 1);

            Pointer kernelParameters = Pointer.to(
                    Pointer.to(d_candidates),
                    Pointer.to(d_results),
                    Pointer.to(d_counter),
                    Pointer.to(new int[]{numPermutations}),
                    Pointer.to(new int[]{MAX_RESULTS})
            );

            int blockSizeX = 256;
            int gridSizeX = (numPermutations + blockSizeX - 1) / blockSizeX;

            JCudaDriver.cuLaunchKernel(function,
                    gridSizeX, 1, 1,
                    blockSizeX, 1, 1,
                    0, null,
                    kernelParameters, null
            );
            JCudaDriver.cuCtxSynchronize();

            int[] countArr = new int[1];
            JCudaDriver.cuMemcpyDtoH(Pointer.to(countArr), d_counter, Sizeof.INT);
            int validFound = Math.min(countArr[0], MAX_RESULTS);

            int[] resultsArr = new int[validFound * 16];
            if (validFound > 0) {
                JCudaDriver.cuMemcpyDtoH(Pointer.to(resultsArr), d_results, (long) validFound * 16 * Sizeof.INT);
            }

            JCudaDriver.cuMemFree(d_candidates);
            JCudaDriver.cuMemFree(d_results);
            JCudaDriver.cuMemFree(d_counter);

            List<int[]> validList = new ArrayList<>();
            for (int i = 0; i < validFound; i++) {
                int[] tile = new int[16];
                System.arraycopy(resultsArr, i * 16, tile, 0, 16);
                validList.add(tile);
            }
            return validList;
        } finally {
            // Pop the context back
            JCudaDriver.cuCtxPopCurrent(new CUcontext());
        }
    }
}
