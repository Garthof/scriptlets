#include "Surf3DFilter.h"

#include <cstdio>

#include "constants.h"
#include "CUDAUtils.h"

surface<void, cudaSurfaceType3D> t_data1;
surface<void, cudaSurfaceType3D> t_data2;


__device__ __forceinline__ float4
kernTexFilterHorOperation4(
        const float4 &val0, const float4 &val1, const float a)
{
    float4 res = val0;

    res.x += a * (val1.x - val0.x);
    res.y += a * (val1.y - val0.y);
    res.z += a * (val1.z - val0.z);
    res.w += a * (val1.w - val0.w);

    return res;
}


__global__ void
kernSurf3DFilterHor1(
        const int depth, const int height, const int width,
        const float a)
{
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int k = threadIdx.y + blockIdx.y * blockDim.y;

    if (j < height && k < depth) {
        float4 prevVal;
        surf3Dread(&prevVal, t_data1, 0*sizeof(float4), j, k);
        surf3Dwrite(prevVal, t_data2, 0*sizeof(float4), j, k);
//        printf("i = 0: %f, %f, %f, %f\n", prevVal.x, prevVal.y, prevVal.z, prevVal.w);
        for (int i = 1; i < width; i++) {
            float4 curVal;
            surf3Dread(&curVal, t_data1, i*sizeof(float4), j, k);
            prevVal = kernTexFilterHorOperation4(curVal, prevVal, a);
            surf3Dwrite(prevVal, t_data2, i*sizeof(float4), j, k);
//            printf("i = %d: %f, %f, %f, %f\n", i, curVal.x, curVal.y, curVal.z, curVal.w);
        }
    }
}


__global__ void
kernSurf3DFilterHor2(
        const int depth, const int height, const int width,
        const float a)
{
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int k = threadIdx.y + blockIdx.y * blockDim.y;

    if (j < height && k < depth) {
        float4 prevVal;
        surf3Dread(&prevVal, t_data2, (width-1)*sizeof(float4), j, k);
        surf3Dwrite(prevVal, t_data1, (width-1)*sizeof(float4), j, k);

        for (int i = width-2; i >= 0; i--) {
            float4 curVal;
            surf3Dread(&curVal, t_data2, i*sizeof(float4), j, k);
            prevVal = kernTexFilterHorOperation4(curVal, prevVal, a);
            surf3Dwrite(prevVal, t_data1, i*sizeof(float4), j, k);
        }
    }
}


void
computeResultFrom3DSurface(
        float4 *const h_resData,
        const float4 *const h_origData,
        const int depth, const int height, const int width,
        const float a)
{
    const int dataSize = depth * height * width;

    // Copy original data in GPU
    float4 *d_origData;
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_origData,
                                 sizeof(*d_origData) * dataSize));

    CUDA_CHECK_RETURN(cudaMemcpy(d_origData, h_origData,
                                 sizeof(*d_origData) * dataSize,
                                 cudaMemcpyHostToDevice));

    // Allocate CUDA arrays
    cudaArray_t a_data1, a_data2;
    const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    const cudaExtent extent = make_cudaExtent(width, height, depth);

    CUDA_CHECK_RETURN(cudaMalloc3DArray(&a_data1, &desc, extent,
                                        cudaArraySurfaceLoadStore));
    CUDA_CHECK_RETURN(cudaMalloc3DArray(&a_data2, &desc, extent,
                                        cudaArraySurfaceLoadStore));

    // Copy data to CUDA array 1
    cudaMemcpy3DParms params1 = { 0 };
    params1.srcPtr = make_cudaPitchedPtr((void *) d_origData,
                                        width * sizeof(*d_origData),
                                        width, height);
    params1.dstArray = a_data1;
    params1.extent = extent;
    params1.kind = cudaMemcpyDeviceToDevice;
    CUDA_CHECK_RETURN(cudaMemcpy3D(&params1));

    // Compute result
    const dim3 blkSize(8, 8);
    const dim3 grdSize(ceil_div(height, (int) blkSize.x),
                       ceil_div(depth, (int) blkSize.y));

    CUDA_CHECK_RETURN(cudaBindSurfaceToArray(t_data1, a_data1, desc));
    CUDA_CHECK_RETURN(cudaBindSurfaceToArray(t_data2, a_data2, desc));

    for (int i = 0; i < NUM_ITERS; i++) {
        kernSurf3DFilterHor1<<<grdSize, blkSize>>>
                           (depth, height, width, a);
        CUDA_CHECK_RETURN(cudaThreadSynchronize());
        CUDA_CHECK_RETURN(cudaGetLastError());

        kernSurf3DFilterHor2<<<grdSize, blkSize>>>
                           (depth, height, width, a);
        CUDA_CHECK_RETURN(cudaThreadSynchronize());
        CUDA_CHECK_RETURN(cudaGetLastError());
    }

    // Copy data from CUDA array 2
    cudaMemcpy3DParms params2 = { 0 };
    params2.srcArray = a_data1;
    params2.dstPtr = make_cudaPitchedPtr((void *) d_origData,
                                        width * sizeof(*d_origData),
                                        width, height);
    params2.extent = extent;
    params2.kind = cudaMemcpyDeviceToDevice;
    CUDA_CHECK_RETURN(cudaMemcpy3D(&params2));

    // Copy result into CPU memory
    CUDA_CHECK_RETURN(cudaMemcpy(h_resData, d_origData,
                                 sizeof(*d_origData) * dataSize,
                                 cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(d_origData));
    CUDA_CHECK_RETURN(cudaFreeArray(a_data1));
    CUDA_CHECK_RETURN(cudaFreeArray(a_data2));
}
