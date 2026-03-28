package dk.roleplay;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import java.util.ArrayList;
import java.util.List;

public class GpuValidator implements CandidateValidator {
    private CUfunction function;
    private final int MAX_RESULTS = 1000;

    public GpuValidator() {
        JCudaDriver.setExceptionsEnabled(true);
        JCudaDriver.cuInit(0);
        CUdevice device = new CUdevice();
        JCudaDriver.cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        JCudaDriver.cuCtxCreate(context, 0, device);

        CUmodule module = new CUmodule();
        JCudaDriver.cuModuleLoad(module, "EternityKernel.ptx");
        function = new CUfunction();
        JCudaDriver.cuModuleGetFunction(function, module, "validateMacroTiles");
    }

    @Override
    public List<int[]> validate(int[] candidates) {
        int numCandidates = candidates.length / 16;
        if (numCandidates == 0) return new ArrayList<>();

        CUdeviceptr d_candidates = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(d_candidates, candidates.length * Sizeof.INT);
        JCudaDriver.cuMemcpyHtoD(d_candidates, Pointer.to(candidates), candidates.length * Sizeof.INT);

        CUdeviceptr d_results = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(d_results, MAX_RESULTS * 16 * Sizeof.INT);
        
        CUdeviceptr d_counter = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(d_counter, Sizeof.INT);
        JCudaDriver.cuMemsetD8(d_counter, (byte) 0, Sizeof.INT);

        Pointer kernelParameters = Pointer.to(
            Pointer.to(d_candidates),
            Pointer.to(d_results),
            Pointer.to(d_counter),
            Pointer.to(new int[]{numCandidates}),
            Pointer.to(new int[]{MAX_RESULTS})
        );

        int blockSizeX = 256;
        int gridSizeX = (numCandidates + blockSizeX - 1) / blockSizeX;

        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, 1, 1,      // Grid dimension
            blockSizeX, 1, 1,     // Block dimension
            0, null,               // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
        JCudaDriver.cuCtxSynchronize();

        int[] counterArr = new int[1];
        JCudaDriver.cuMemcpyDtoH(Pointer.to(counterArr), d_counter, Sizeof.INT);
        int validFound = Math.min(counterArr[0], MAX_RESULTS);

        int[] resultsArr = new int[validFound * 16];
        JCudaDriver.cuMemcpyDtoH(Pointer.to(resultsArr), d_results, validFound * 16 * Sizeof.INT);

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
    }
}
