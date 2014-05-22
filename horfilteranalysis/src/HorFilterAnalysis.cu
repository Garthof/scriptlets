#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <vector>

const int WORK_SIZE = 145;
const int NUM_ITERS = 100;

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }


template <typename T>
inline T
ceil_div(const T &a, const T &b)
{
    return ((a + b - 1) / b);
}


__global__ void
kernFilterHor(
        float *const data,
        const int depth, const int height, const int width, const int nn,
        const float a)
{
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int k = threadIdx.y + blockIdx.y * blockDim.y;

    if (j < height && k < depth) {
        for (int i = 1; i < width; i++) {
            const int basePos0 = (i + j * width + k * width * height) * nn;
            const int basePos1 = ((i-1) + j * width + k * width * height) * nn;

            for (int n = 0; n < nn; n++) {
                const int pos0 = basePos0 + n;
                const int pos1 = basePos1 + n;

                data[pos0] += a * (data[pos1] - data[pos0]);
            }
        }

        for (int i = width-2; i >= 0; i--) {
            const int basePos0 = (i + j * width + k * width * height) * nn;
            const int basePos1 = ((i+1) + j * width + k * width * height) * nn;

            for (int n = 0; n < nn; n++) {
                const int pos0 = basePos0 + n;
                const int pos1 = basePos1 + n;

                data[pos0] += a * (data[pos1] - data[pos0]);
            }
        }
    }
}


__global__ void
kernFilterHor3(
        float3 *const data,
        const int depth, const int height, const int width,
        const float a)
{
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int k = threadIdx.y + blockIdx.y * blockDim.y;

    if (j < height && k < depth) {
        for (int i = 1; i < width; i++) {
            const int pos0 = (i + j * width + k * width * height);
            const int pos1 = ((i-1) + j * width + k * width * height);

            const float3 valPos0 = data[pos0];
            const float3 valPos1 = data[pos1];
            float3 res = valPos0;

            res.x += a * (valPos1.x - valPos0.x);
            res.y += a * (valPos1.y - valPos0.y);
            res.z += a * (valPos1.z - valPos0.z);

            data[pos0] = res;
        }

        for (int i = width-2; i >= 0; i--) {
            const int pos0 = (i + j * width + k * width * height);
            const int pos1 = ((i+1) + j * width + k * width * height);

            const float3 valPos0 = data[pos0];
            const float3 valPos1 = data[pos1];
            float3 res = valPos0;

            res.x += a * (valPos1.x - valPos0.x);
            res.y += a * (valPos1.y - valPos0.y);
            res.z += a * (valPos1.z - valPos0.z);

            data[pos0] = res;
        }
    }
}


__global__ void
kernFilterHor4(
        float4 *const data,
        const int depth, const int height, const int width,
        const float a)
{
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int k = threadIdx.y + blockIdx.y * blockDim.y;

    if (j < height && k < depth) {
        for (int i = 1; i < width; i++) {
            const int pos0 = (i + j * width + k * width * height);
            const int pos1 = ((i-1) + j * width + k * width * height);

            const float4 valPos0 = data[pos0];
            const float4 valPos1 = data[pos1];
            float4 res = valPos0;

            res.x += a * (valPos1.x - valPos0.x);
            res.y += a * (valPos1.y - valPos0.y);
            res.z += a * (valPos1.z - valPos0.z);
            res.w += a * (valPos1.w - valPos0.w);

            data[pos0] = res;
        }

        for (int i = width-2; i >= 0; i--) {
            const int pos0 = (i + j * width + k * width * height);
            const int pos1 = ((i+1) + j * width + k * width * height);

            const float4 valPos0 = data[pos0];
            const float4 valPos1 = data[pos1];
            float4 res = valPos0;

            res.x += a * (valPos1.x - valPos0.x);
            res.y += a * (valPos1.y - valPos0.y);
            res.z += a * (valPos1.z - valPos0.z);
            res.w += a * (valPos1.w - valPos0.w);

            data[pos0] = res;
        }
    }
}


void
computeRefResult(
        float *const h_resData,
        const float *const h_origData,
        const int depth, const int height, const int width, const int nn,
        const float a)
{
    const int dataSize = depth * height * width * nn;

    // Copy original data in GPU
    float *d_data;
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


void
computeResult3(
        float3 *const h_resData,
        const float3 *const h_origData,
        const int depth, const int height, const int width,
        const float a)
{
    const int dataSize = depth * height * width;

    // Copy original data in GPU
    float3 *d_data;
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
        kernFilterHor3<<<grdSize, blkSize>>>(d_data, depth, height, width, a);
    }

    CUDA_CHECK_RETURN(cudaThreadSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());

    // Copy result into CPU memory
    CUDA_CHECK_RETURN(cudaMemcpy(h_resData, d_data,
                                 sizeof(*d_data) * dataSize,
                                 cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(d_data));
}



void
computeResult4(
        float4 *const h_resData,
        const float4 *const h_origData,
        const int depth, const int height, const int width,
        const float a)
{
    const int dataSize = depth * height * width;

    // Copy original data in GPU
    float4 *d_data;
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
        kernFilterHor4<<<grdSize, blkSize>>>(d_data, depth, height, width, a);
    }

    CUDA_CHECK_RETURN(cudaThreadSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());

    // Copy result into CPU memory
    CUDA_CHECK_RETURN(cudaMemcpy(h_resData, d_data,
                                 sizeof(*d_data) * dataSize,
                                 cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(d_data));
}


bool
compareResults(const std::vector<float> a, const std::vector<float> b)
{
    assert(a.size() == b.size());

    for (int i = 0; i < a.size(); i++) {
        if (std::abs(a[i] - b[i]) > 0.001f) return false;
    }

    return true;
}



int main(void) {
    const float sigma = 0.5f;
    const float a = std::exp(-std::sqrt(2.f) / sigma);
    const int depth = WORK_SIZE;
    const int height = WORK_SIZE;
    const int width = WORK_SIZE;
    const int nn = 4;
    const int dataSize = depth * height * width * nn;

    // Generate data in CPU
    std::vector<float> h_origData(dataSize);
    for (int i = 0; i < dataSize; i++) {
        const float r = float(std::rand()) / float(RAND_MAX);
        h_origData[i] = r - 0.5f;
    }

    // Copy data into GPU
    std::vector<float> h_resData(dataSize);
    computeRefResult(&h_resData[0], &h_origData[0], depth, height, width, nn, a);

    if (nn == 3)  {
        const std::vector<float> h_origData3 = h_origData;
        std::vector<float> h_resData3(dataSize);
        computeResult3((float3 *) &h_resData3[0], (float3 *) &h_origData3[0],
                       depth, height, width, a);

        if (compareResults(h_resData, h_resData3)) {
            std::cout << "Results are equal" << std::endl;
        } else {
            std::cerr << "Results are different" << std::endl;
        }
    }

    if (nn == 4)  {
        const std::vector<float> h_origData4 = h_origData;
        std::vector<float> h_resData4(dataSize);
        computeResult4((float4 *) &h_resData4[0], (float4 *) &h_origData4[0],
                       depth, height, width, a);

        if (compareResults(h_resData, h_resData4)) {
            std::cout << "Results are equal" << std::endl;
        } else {
            std::cerr << "Results are different" << std::endl;
        }
    }

    CUDA_CHECK_RETURN(cudaDeviceReset());

    return 0;
}
