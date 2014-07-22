#include "CUDAUtils.h"

#include <cstdio>

template<typename T>
__device__ __forceinline__ void
kernFilterHorOperation(
        T *const data,
        const int basePos0, const int basePos1,
        const int nn, const float a)
{
    const int channelPos0 = basePos0 * nn;
    const int channelPos1 = basePos1 * nn;

    for (int n = 0; n < nn; n++) {
        const int pos0 = channelPos0 + n;
        const int pos1 = channelPos1 + n;

        data[pos0] += a * (data[pos1] - data[pos0]);
    }
}


template<>
__device__ __forceinline__ void
kernFilterHorOperation(
        float2 *const data,
        const int basePos0, const int basePos1,
        const int nn, const float a)
{
    const float2 valPos0 = data[basePos0];
    const float2 valPos1 = data[basePos1];
    float2 res = valPos0;

    res.x += a * (valPos1.x - valPos0.x);
    res.y += a * (valPos1.y - valPos0.y);

    data[basePos0] = res;
}


template<>
__device__ __forceinline__ void
kernFilterHorOperation(
        float3 *const data,
        const int basePos0, const int basePos1,
        const int nn, const float a)
{
    const float3 valPos0 = data[basePos0];
    const float3 valPos1 = data[basePos1];
    float3 res = valPos0;

    res.x += a * (valPos1.x - valPos0.x);
    res.y += a * (valPos1.y - valPos0.y);
    res.z += a * (valPos1.z - valPos0.z);

    data[basePos0] = res;
}


template<>
__device__ __forceinline__ void
kernFilterHorOperation(
        float4 *const data,
        const int basePos0, const int basePos1,
        const int nn, const float a)
{
    const float4 valPos0 = data[basePos0];
    const float4 valPos1 = data[basePos1];
    float4 res = valPos0;

    res.x += a * (valPos1.x - valPos0.x);
    res.y += a * (valPos1.y - valPos0.y);
    res.z += a * (valPos1.z - valPos0.z);
    res.w += a * (valPos1.w - valPos0.w);

    data[basePos0] = res;
}


template<typename T>
__global__ void
kernFilterHor(
        T *const data,
        const int depth, const int height, const int width, const int nn,
        const float a)
{
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int k = threadIdx.y + blockIdx.y * blockDim.y;

    if (j < height && k < depth) {
        for (int i = 1; i < width; i++) {
            const int pos0 = (i + j * width + k * width * height);
            const int pos1 = ((i-1) + j * width + k * width * height);

            kernFilterHorOperation(data, pos0, pos1, nn, a);
        }

        for (int i = width-2; i >= 0; i--) {
            const int pos0 = (i + j * width + k * width * height);
            const int pos1 = ((i+1) + j * width + k * width * height);

            kernFilterHorOperation(data, pos0, pos1, nn, a);
        }
    }
}


template<typename T>
void
computeResult(
        T *const h_resData,
        const T *const h_origData,
        const int depth, const int height, const int width, const int nn,
        const float a)
{
    const int dataSize = depth * height * width * nn;

    // Copy original data in GPU
    T *d_data;
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_data,
                                 sizeof(*d_data) * dataSize));

    CUDA_CHECK_RETURN(cudaMemcpy(d_data, h_origData,
                                 sizeof(*d_data) * dataSize,
                                 cudaMemcpyHostToDevice));

    // Compute result
    const dim3 blkSize(8, 8);
    const dim3 grdSize(ceil_div(height, (int) blkSize.x),
                       ceil_div(depth, (int) blkSize.y));

    for (int i = 0; i < NUM_ITERS; i++) {
        kernFilterHor<<<grdSize, blkSize>>>(d_data, depth, height, width, nn, a);
    }

    CUDA_CHECK_RETURN(cudaThreadSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());

    // Copy result into CPU memory
    CUDA_CHECK_RETURN(cudaMemcpy(h_resData, d_data,
                                 sizeof(*d_data) * dataSize,
                                 cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(d_data));
}
