#include "cudapca.h"

#include <cstdio>
#include <cstdlib>

#include "cublas_v2.h"

#include <thrust/device_vector.h>

// Auxiliary functions

#ifdef CUDAPCA_USE_FLOAT
#define cublasXaxpy cublasSaxpy
#define cublasXdot cublasSdot
#define cublasXgemm cublasSgemm
#define cublasXgemv cublasSgemv
#define cublasXger cublasSger
#define cublasXscal cublasSscal
#else
#define cublasXaxpy cublasDaxpy
#define cublasXdot cublasDdot
#define cublasXgemm cublasDgemm
#define cublasXgemv cublasDgemv
#define cublasXger cublasDger
#define cublasXscal cublasDscal
#endif


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
    thrust::device_vector<data_t> d_mean(d_sum);
    alpha = 1.f / dataSize;

    cublasCheck(cublasXscal(handle, patchSize, &alpha,
                            d_mean.data().get(), 1));

//#define TEST_EIGEN_MEAN
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

    // Compute the product of all patches with its transpose. This
    // is equivalent a computing the product of all random variables
    // (each patch is a sample of patchSize random variables). The
    // resulting matrix contains in cell (i,j) the sum of the products
    // of the elements i and j of all patches.
    thrust::device_vector<data_t> d_prod(patchSize * patchSize);
    alpha = 1.f;
    beta = 0.f;

    cublasCheck(cublasXgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                            patchSize, patchSize, dataSize,
                            &alpha,
                            d_patches.data, patchSize,
                            d_patches.data, patchSize,
                            &beta,
                            d_prod.data().get(), patchSize));

//#define TEST_EIGEN_PROD
#ifdef TEST_EIGEN_PROD
    // Get values of product and return them
    data_t *d_prod_copy;

    cudaCheck(cudaMalloc((void**) &d_prod_copy,
                         patchSize * patchSize * sizeof(*d_prod_copy)));

    cudaCheck(cudaMemcpy(d_prod_copy, d_prod.data().get(),
                         patchSize * patchSize * sizeof(*d_prod_copy),
                         cudaMemcpyHostToDevice));

    return CUDAPCAData(1, patchSize, patchSize, d_prod_copy);
#endif

    // Compute the covariance matrix. This is done in several steps.
    // 1.   Compute the mean-patch product matrix, i.e., the matrix
    //      resulting of multiplying the mean patch by itself. This
    //      is a symmetric matrix (its transpose is the same) so the
    //      major order used by CUBLAS does not matter. I use the CUBLAS
    //      ger function instead of the spr one because the latter returns
    //      a packed matrix, and I want a full-fledged one. This mean
    //      product is divided by the number of occurrences (the number
    //      of elements in the volume).
    thrust::device_vector<data_t> d_meanprod(patchSize * patchSize, 0.f);
    alpha = 1.f / dataSize;

    cublasCheck(cublasXger(handle, patchSize, patchSize,
                           &alpha,
                           d_mean.data().get(), 1,
                           d_mean.data().get(), 1,
                           d_meanprod.data().get(), patchSize));

//#define TEST_EIGEN_MEANPROD
#ifdef TEST_EIGEN_MEANPROD
    // Get values of mean product and return them
    data_t *d_meanprod_copy;

    cudaCheck(cudaMalloc((void**) &d_meanprod_copy,
                         patchSize * patchSize * sizeof(*d_meanprod_copy)));

    cudaCheck(cudaMemcpy(d_meanprod_copy, d_meanprod.data().get(),
                         patchSize * patchSize * sizeof(*d_meanprod_copy),
                         cudaMemcpyHostToDevice));

    return CUDAPCAData(1, patchSize, patchSize, d_meanprod_copy);
#endif

    // 2.   Withdraw the mean product matrix from the patch product
    //      matrix.
    thrust::device_vector<data_t> d_cov(d_prod);
    alpha = -1.f;

    cublasCheck(cublasXaxpy(handle, patchSize * patchSize, &alpha,
                            d_meanprod.data().get(), 1,
                            d_cov.data().get(), 1));

    // 3.   Normalize the result.
    alpha = 1.f / dataSize;

    cublasCheck(cublasXscal(handle, patchSize * patchSize, &alpha,
                            d_cov.data().get(), 1));

//#define TEST_EIGEN_COV
#ifdef TEST_EIGEN_COV
    // Get values of covariance and return them
    data_t *d_cov_copy;

    cudaCheck(cudaMalloc((void**) &d_cov_copy,
            patchSize * patchSize * sizeof(*d_cov_copy)));

    cudaCheck(cudaMemcpy(d_cov_copy, d_cov.data().get(),
            patchSize * patchSize * sizeof(*d_cov_copy),
            cudaMemcpyHostToDevice));

    return CUDAPCAData(1, patchSize, patchSize, d_cov_copy);
#endif

    // Compute eigenvectors from the covariance matrix. The approach used
    // here is an iterative one. It is based on the code of Adams et al.
    // "Gaussian KD-Trees for Fast High-Dimensional Filtering" (2009).
    // Consider each row in the covariance matrix a first approximation
    // of each eigenvector. The code proceeds iteratively in several steps.
    thrust::device_vector<data_t> d_eigenvecs(d_cov);

    while(true) {
        // 1. Find an orthogonal base for the current eigenvectors
        for (int i = 0; i < patchSize; i++) {
            // Make vector i independent of all previous ones
            for (int j = 0; j < i; j++) {
                data_t dot = 0.f;

                // Compute dot = v_i * v_j, where * is the dot product
                cublasCheck(cublasXdot(handle, patchSize,
                                       d_eigenvecs.data().get() + i*patchSize, 1,
                                       d_eigenvecs.data().get() + j*patchSize, 1,
                                       &dot));

                // Compute v_i = -dot * v_j + v_i
                dot = -dot;
                cublasCheck(cublasXaxpy(handle, patchSize, &dot,
                                        d_eigenvecs.data().get() + j*patchSize, 1,
                                        d_eigenvecs.data().get() + i*patchSize, 1));
            }
        }

        break;      // Test
    }

#define TEST_EIGEN_COMP
#ifdef TEST_EIGEN_COMP
    // Get values of eigenvectors and return them
    data_t *d_eigenvecs_copy;

    cudaCheck(cudaMalloc((void**) &d_eigenvecs_copy,
            patchSize * patchSize * sizeof(*d_eigenvecs_copy)));

    cudaCheck(cudaMemcpy(d_eigenvecs_copy, d_eigenvecs.data().get(),
            patchSize * patchSize * sizeof(*d_eigenvecs_copy),
            cudaMemcpyHostToDevice));

    return CUDAPCAData(1, patchSize, patchSize, d_eigenvecs_copy);
#endif

    // Clean and return
    cublasCheck(cublasDestroy(handle));
    return CUDAPCAData(0, 0, 0, 0);
}
