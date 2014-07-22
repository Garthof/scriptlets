#include "Tex3DFilter.h"

#include <cstdio>

#include "constants.h"
#include "CUDAUtils.h"

// Constructs a 3D texture where each element is a float4. By default,
// texture coordinates are not normalized, no interpolation is done to
// fetch the values (filter mode is point), and the addressing mode is
// clamp.
texture<float4, cudaTextureType3D, cudaReadModeElementType> t_data3D;
texture<float4, cudaTextureType1D> t_data;


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
kernTex3DFilterHor1(
        float4 *const g_resData,
        const int depth, const int height, const int width,
        const float a)
{
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int k = threadIdx.y + blockIdx.y * blockDim.y;

    if (j < height && k < depth) {
        // Note on coordinates: using point filter mode is equivalent to a
        // nearest point sampling. By CUDA's specification, this means that
        // tex1D(x) == T[(int) floor(x)]. To achieve a correct conversion
        // during flooring, it is convenient to use tex1D(x+.5f) in the code.
        float4 prevVal;
        const int base = j * width + k * width * height;
        g_resData[base+0] = prevVal = tex3D(t_data3D, 0+.5f, j+.5f, k+.5f);
//        printf("i = 0: %f, %f, %f, %f\n", prevVal.x, prevVal.y, prevVal.z, prevVal.w);
        for (int i = 1; i < width; i++) {
            const float4 curVal = tex3D(t_data3D, i+.5f, j+.5f, k+.5f);
            const float4 newVal = kernTexFilterHorOperation4(curVal, prevVal, a);
            g_resData[base+i] = prevVal = newVal;
//            printf("i = %d: %f, %f, %f, %f\n", i, curVal.x, curVal.y, curVal.z, curVal.w);
        }
    }
}


__global__ void
kernTex3DFilterHor2(
        float4 *const g_resData,
        const int depth, const int height, const int width,
        const float a)
{
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int k = threadIdx.y + blockIdx.y * blockDim.y;

    if (j < height && k < depth) {
        float4 prevVal;
        const int base = j * width + k * width * height;
        g_resData[base+(width-1)] = prevVal = tex1Dfetch(t_data, base+(width-1));

        for (int i = width-2; i >= 0; i--) {
            const float4 curVal = tex1Dfetch(t_data, base+i);
            const float4 newVal = kernTexFilterHorOperation4(curVal, prevVal, a);
            g_resData[base+i] = prevVal = newVal;
        }
    }
}


void
computeResultFrom3DTexture(
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

    // Allocate CUDA array to store original data
    cudaArray_t a_origData;
    const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    const cudaExtent extent = make_cudaExtent(width, height, depth);

    CUDA_CHECK_RETURN(cudaMalloc3DArray(&a_origData, &desc, extent));

    // Copy data to CUDA array
    cudaMemcpy3DParms params = { 0 };
    params.srcPtr = make_cudaPitchedPtr((void *) d_origData,
                                        width * sizeof(*d_origData),
                                        width, height);
    params.dstArray = a_origData;
    params.extent = extent;
    params.kind = cudaMemcpyDeviceToDevice;
    CUDA_CHECK_RETURN(cudaMemcpy3D(&params));

    // Generate buffer to store result in GPU
    float4 *d_resData;
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_resData,
                                 sizeof(*d_resData) * dataSize));

    // Compute result
    const dim3 blkSize(8, 8);
    const dim3 grdSize(ceil_div(height, (int) blkSize.x),
                       ceil_div(depth, (int) blkSize.y));

    for (int i = 0; i < NUM_ITERS; i++) {
        CUDA_CHECK_RETURN(cudaBindTextureToArray(t_data3D, a_origData, desc));
        kernTex3DFilterHor1<<<grdSize, blkSize>>>
                           (d_resData, depth, height, width, a);
        CUDA_CHECK_RETURN(cudaThreadSynchronize());
        CUDA_CHECK_RETURN(cudaGetLastError());
        CUDA_CHECK_RETURN(cudaUnbindTexture(t_data3D));

        CUDA_CHECK_RETURN(cudaBindTexture(0, t_data, d_resData,
                                          sizeof(*d_resData) * dataSize));
        kernTex3DFilterHor2<<<grdSize, blkSize>>>
                           (d_origData, depth, height, width, a);
        CUDA_CHECK_RETURN(cudaThreadSynchronize());
        CUDA_CHECK_RETURN(cudaGetLastError());
        CUDA_CHECK_RETURN(cudaUnbindTexture(t_data));
    }

    // Copy result into CPU memory
    CUDA_CHECK_RETURN(cudaMemcpy(h_resData, d_origData,
                                 sizeof(*d_origData) * dataSize,
                                 cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(d_origData));
    CUDA_CHECK_RETURN(cudaFree(d_resData));
    CUDA_CHECK_RETURN(cudaFreeArray(a_origData));
}
