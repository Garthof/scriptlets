#include "cudapca.h"

#include <cstdio>
#include <cstdlib>

#include "cublas_v2.h"

#include <thrust/device_vector.h>

// Auxiliary functions

#define cudaCheck(call)                                                        \
{                                                                              \
    const cudaError_t stat = call;                                             \
    if (call != cudaSuccess) {                                                 \
        fprintf(stderr, "Error %s at line %d in file %s\n",                    \
                cudaGetErrorString(stat), __LINE__, __FILE__);                 \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}


#define cublasCheck(call)                                                      \
{                                                                              \
    const cublasStatus_t stat = call;                                          \
    if (stat != CUBLAS_STATUS_SUCCESS) {                                       \
        fprintf(stderr, "CUBLAS error %d at line %d in file %s\n",             \
                stat, __LINE__, __FILE__);                                     \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}


inline int
ceilDiv(const int a, const int b)
{
    return ((a + b - 1) / b);
}


// CUDA kernels

__global__ void
kernGeneratePatches(
        CUDAPCA::data_t *const g_patchData,
        const CUDAPCA::data_t *const g_data,
        const int depth, const int height, const int width,
        const int patchRadius)
{
    // Get and check thread index
    const int dataSize = depth * height * width;
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= dataSize) return;

    // Get position in the volume of the center of the patch
    const int ck = idx / (width * height);
    const int cj = (idx - ck * width * height) / width;
    const int ci = idx % width;

    // Build patch. Values outside edges are nullified
    const int patchDiam = (2 * patchRadius + 1);
    const int patchSize = patchDiam * patchDiam * patchDiam;
    CUDAPCA::data_t *const g_patchDataLocal = g_patchData + idx * patchSize;
    int cont = 0;

    for (int k = -patchRadius; k <= +patchRadius; k++) {
        for (int j = -patchRadius; j <= +patchRadius; j++) {
            for (int i = -patchRadius; i <= +patchRadius; i++) {
                const int pk = ck + k;
                const int pj = cj + j;
                const int pi = ci + i;

                if (0 <= pk && pk < depth
                        && 0 <= pj && pj < height
                        && 0 <= pi && pi < width) {

                    const int pos = pi + pj * width + pk * width * height;
                    g_patchDataLocal[cont++] = g_data[pos];
                } else {
                    g_patchDataLocal[cont++] = 0;
                }
            }
        }
    }
}


// Class CUDAPCAData

CUDAPCA::CUDAPCAData::CUDAPCAData(
        const int _depth,
        const int _height,
        const int _width,
        const CUDAPCA::data_t *const _data)
    : depth(_depth)
    , height(_height)
    , width(_width)
    , data(_data)
{

}


CUDAPCA::CUDAPCAData::~CUDAPCAData() {
    printf("Releasing GPU memory...\n");
    void *cData = const_cast<void *>(static_cast<const void *>(data));
    cudaCheck(cudaFree(cData));
}


// Class CUDAPCAPatches

CUDAPCA::CUDAPCAPatches::CUDAPCAPatches(
        const int _depth,
        const int _height,
        const int _width,
        const int _patchRadius, const CUDAPCA::data_t *const _data)
    : CUDAPCAData(_depth, _height, _width, _data)
    , patchRadius(_patchRadius)
{

}


// Namespace functions

CUDAPCA::CUDAPCAData
CUDAPCA::uploadData(
        const int depth,
        const int height,
        const int width,
        const void *const h_data)
{
    const int dataSize = depth * height * width;
    data_t *d_data;

    // Allocate and copy data to GPU memory
    cudaCheck(cudaMalloc((void**) &d_data,
                         dataSize * sizeof(*d_data)));

    cudaCheck(cudaMemcpy(d_data, h_data,
                         dataSize * sizeof(*d_data),
                         cudaMemcpyHostToDevice));

    // Generate CUDAPCAData object and return
    CUDAPCAData h_pcaData(depth, width, height, d_data);
    return h_pcaData;
}


std::auto_ptr<CUDAPCA::data_t>
CUDAPCA::downloadData(const CUDAPCA::CUDAPCAData &d_data)
{
    const int dataSize = d_data.depth * d_data.height * d_data.width;
    data_t *h_data;

    // Allocate space in host and copy data from GPU memory
    cudaCheck(cudaMallocHost((void **) &h_data,
                             dataSize * sizeof(*h_data)));

    cudaCheck(cudaMemcpy(h_data, d_data.data,
                         dataSize * sizeof(*h_data),
                         cudaMemcpyDeviceToHost));

    return std::auto_ptr<data_t>(h_data);
}


CUDAPCA::CUDAPCAPatches
CUDAPCA::generatePatches(
        const CUDAPCA::CUDAPCAData &d_data,
        const int patchRadius)
{
    // Allocate patch space in GPU memory
    const int patchDiam = 2 * patchRadius + 1;
    const int patchSize = patchDiam * patchDiam * patchDiam;
    const int dataSize = d_data.depth * d_data.height * d_data.width;

    data_t *d_patchData;
    cudaCheck(cudaMalloc((void **) &d_patchData,
                               patchSize * dataSize * sizeof(*d_patchData)));

    // Launch kernel and wait to finish
    const dim3 dimBlk(512);
    const dim3 dimGrd(ceilDiv(dataSize, dimBlk.x));

    kernGeneratePatches<<<dimGrd, dimBlk>>>(
            d_patchData, d_data.data,
            d_data.depth, d_data.width, d_data.height,
            patchRadius);

    cudaCheck(cudaThreadSynchronize());
    cudaCheck(cudaGetLastError());

    return CUDAPCAPatches(
            d_data.depth, d_data.width, d_data.height,
            patchRadius, d_patchData);
}


std::auto_ptr<CUDAPCA::data_t>
CUDAPCA::downloadPatches(const CUDAPCA::CUDAPCAPatches &d_patches)
{
    const int patchDiam = 2 * d_patches.patchRadius + 1;
    const int patchSize = patchDiam * patchDiam * patchDiam;
    const int patchesSize = d_patches.depth * d_patches.height
                            * d_patches.width * patchSize;
    data_t *h_patches;

    // Allocate space in host and copy data from GPU memory
    cudaCheck(cudaMallocHost((void **) &h_patches,
                             patchesSize * sizeof(*h_patches)));

    cudaCheck(cudaMemcpy(h_patches, d_patches.data,
                         patchesSize * sizeof(*h_patches),
                         cudaMemcpyDeviceToHost));

    return std::auto_ptr<data_t>(h_patches);
}


CUDAPCA::CUDAPCAData
CUDAPCA::generateEigenvecs(const CUDAPCA::CUDAPCAPatches &d_patches)
{
    // Create CUBLAS handle
    cublasHandle_t handle;
    cublasCheck(cublasCreate(&handle));

    // Compute main of all patches.
    // Generate device vector with ones.
    const int dataSize = d_patches.depth * d_patches.height * d_patches.width;
    const thrust::device_vector<data_t> d_ones(dataSize, 1.f);

    // Multiply patch space by device vector in order to sum all the
    // patches in a component-wise function. CUBLAS gemv function
    // computes y = aAx + by, and expects the matrix A to be in
    // column-major format. As currently each row is a single patch,
    // I do not need transpose the matrix.
    const int patchDiam = (2 * d_patches.patchRadius + 1);
    const int patchSize = patchDiam * patchDiam * patchDiam;
    data_t alpha = 1;
    data_t beta = 0;

    thrust::device_vector<data_t> d_sum(patchSize);

#ifdef CUDAPCA_USE_FLOAT
#define cublasXgemv cublasSgemv
#else
#define cublasXgemv cublasDgemv
#endif

    cublasCheck(cublasXgemv(handle, CUBLAS_OP_N,
                            patchSize, dataSize,
                            &alpha,
                            d_patches.data, patchSize,
                            d_ones.data().get(), 1,
                            &beta,
                            d_sum.data().get(), 1));

//#define TEST_EIGEN_SUM
#ifdef TEST_EIGEN_SUM
    // Get values of sum and return them
    data_t *d_sum_copy;

    cudaCheck(cudaMalloc((void**) &d_sum_copy,
                         patchSize * sizeof(*d_sum_copy)));

    cudaCheck(cudaMemcpy(d_sum_copy, d_sum.data().get(),
                         patchSize * sizeof(*d_sum_copy),
                         cudaMemcpyHostToDevice));

    return CUDAPCAData(1, 1, patchSize, d_sum_copy);
#endif

    // Divide the sum vector between the number of samples to
    // get the mean patch
    thrust::device_vector<data_t> d_mean(patchSize, 0.f);
    alpha = 1.f/dataSize;

#ifdef CUDAPCA_USE_FLOAT
#define cublasXaxpy cublasSaxpy
#else
#define cublasXaxpy cublasDaxpy
#endif

    cublasCheck(cublasXaxpy(handle, patchSize, &alpha,
                            d_sum.data().get(), 1,
                            d_mean.data().get(), 1));

#define TEST_EIGEN_MEAN
#ifdef TEST_EIGEN_MEAN
    // Get values of mean and return them
    data_t *d_mean_copy;

    cudaCheck(cudaMalloc((void**) &d_mean_copy,
                         patchSize * sizeof(*d_mean_copy)));

    cudaCheck(cudaMemcpy(d_mean_copy, d_mean.data().get(),
                         patchSize * sizeof(*d_mean_copy),
                         cudaMemcpyHostToDevice));

    return CUDAPCAData(1, 1, patchSize, d_mean_copy);
#endif

    // Clean and return
    cublasCheck(cublasDestroy(handle));
    return CUDAPCAData(0, 0, 0, 0);
}
