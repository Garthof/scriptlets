#include "cudapca.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "cublas_v2.h"

#include <thrust/device_vector.h>

#include "cudadebug.h"

// Auxiliary functions

#ifdef CUDAPCA_USE_FLOAT
#define cublasXaxpy cublasSaxpy
#define cublasXdot cublasSdot
#define cublasXgeam cublasSgeam
#define cublasXgemm cublasSgemm
#define cublasXgemv cublasSgemv
#define cublasXger cublasSger
#define cublasXnrm2 cublasSnrm2
#define cublasXscal cublasSscal
#else
#define cublasXaxpy cublasDaxpy
#define cublasXdot cublasDdot
#define cublasXgeam cublasDgeam
#define cublasXgemm cublasDgemm
#define cublasXgemv cublasDgemv
#define cublasXger cublasDger
#define cublasXnrm2 cublasDnrm2
#define cublasXscal cublasDscal
#endif


#define cudaCheck(call)                                                        \
{                                                                              \
    const cudaError_t stat = call;                                             \
    if (stat != cudaSuccess) {                                                 \
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
        const data_t *const _data)
    : depth(_depth)
    , height(_height)
    , width(_width)
    , data(0)
{
    init(depth * height * width, _data);
}


CUDAPCA::CUDAPCAData::CUDAPCAData(
        const int _depth,
        const int _height,
        const int _width,
        const int dataSize,
        const data_t *const _data)
    : depth(_depth)
    , height(_height)
    , width(_width)
    , data(0)
{
    init(dataSize, _data);
}


CUDAPCA::CUDAPCAData::~CUDAPCAData() {
    printf("Releasing GPU memory...");
    cudaCheck(cudaFree(const_cast<data_t *>(data)));
}


void
CUDAPCA::CUDAPCAData::init(
        const int dataSize,
        const data_t *const _data)
{
    cudaCheck(cudaMalloc(reinterpret_cast<void **>(const_cast<data_t **>(&data)),
                         dataSize * sizeof(*data)));

    cudaCheck(cudaMemcpy(const_cast<data_t *>(data), _data,
                         dataSize * sizeof(*data),
                         cudaMemcpyDeviceToDevice));
}


// Class CUDAPCAPatches

CUDAPCA::CUDAPCAPatches::CUDAPCAPatches(
        const int _depth,
        const int _height,
        const int _width,
        const int _patchRadius,
        const data_t *const _data)
    : CUDAPCAData(_depth, _height, _width,
                  _depth * _height * _width * std::pow(2*_patchRadius+1, 3),
                  _data)
    , patchRadius(_patchRadius)
{

}


// Namespace functions

CUDAPCA::CUDAPCAData
CUDAPCA::uploadData(
        const int depth,
        const int height,
        const int width,
        const std::vector<data_t> &h_data)
{
    const int dataSize = depth * height * width;
    data_t *d_data;

    // Allocate and copy data to GPU memory
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_data),
                         dataSize * sizeof(*d_data)));

    cudaCheck(cudaMemcpy(d_data, h_data.data(),
                         dataSize * sizeof(*d_data),
                         cudaMemcpyHostToDevice));

    // Generate CUDAPCAData object, clean and return
    CUDAPCAData h_pcaData(depth, width, height, d_data);

    cudaCheck(cudaFree(d_data));

    return h_pcaData;
}


std::vector<CUDAPCA::data_t>
CUDAPCA::downloadData(const CUDAPCA::CUDAPCAData &d_data)
{
    const int dataSize = d_data.depth * d_data.height * d_data.width;

    std::vector<data_t> h_data(dataSize);

    cudaCheck(cudaMemcpy(h_data.data(), d_data.data,
                         dataSize * sizeof(*d_data.data),
                         cudaMemcpyDeviceToHost));

    return h_data;
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

    CUDAPCAPatches patches(
            d_data.depth, d_data.width, d_data.height,
            patchRadius, d_patchData);

    cudaCheck(cudaFree(d_patchData));

    return patches;
}


std::vector<CUDAPCA::data_t>
CUDAPCA::downloadPatches(const CUDAPCA::CUDAPCAPatches &d_patches)
{
    const int patchDiam = 2 * d_patches.patchRadius + 1;
    const int patchSize = patchDiam * patchDiam * patchDiam;
    const int patchesSize = d_patches.depth * d_patches.height
                            * d_patches.width * patchSize;

    std::vector<data_t> h_patches(patchesSize);

    cudaCheck(cudaMemcpy(h_patches.data(), d_patches.data,
                         patchesSize * sizeof(*d_patches.data),
                         cudaMemcpyDeviceToHost));

    return h_patches;
}


CUDAPCA::CUDAPCAData
CUDAPCA::generateEigenvecs(
        const CUDAPCA::CUDAPCAPatches &d_patches,
        const int numPCADims)
{
    // Create CUBLAS handle
    cublasHandle_t handle;
    cublasCheck(cublasCreate(&handle));

    // Each patch is considered as a sample of as many random variables as
    // the patch size. Each patch is stored as a row of the d_patches matrix,
    // and each column corresponds to a distinct random variable.
    const int dataSize = d_patches.depth * d_patches.height * d_patches.width;
    const int patchDiam = (2 * d_patches.patchRadius + 1);
    const int patchSize = patchDiam * patchDiam * patchDiam;

    // Compute the mean patch, in several steps:
    //
    // 1.   Multiply the columns of patch matrix space by the one-vector.
    //      For each column/random variable, the sum of the samples is
    //      obtained.
    const thrust::device_vector<data_t> d_ones(dataSize, 1.f);
    data_t alpha = 1.f;
    data_t beta = 0.f;


    // Note: BLAS gemv function computes y = aAx + by. All BLAS functions
    // expect matrices to be stored in column-major format (a la Fortran).
    // A rows correspond to single patches, I do not need transpose the
    // matrix.
    thrust::device_vector<data_t> d_sum(patchSize, 0.f);

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
                         cudaMemcpyDeviceToDevice));

    return CUDAPCAData(1, 1, patchSize, d_sum_copy);
#endif

    // 2.   Divide the sum vector between the number of samples to
    //      get the mean patch. BLAS scal function computes x = ax.
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
                         cudaMemcpyDeviceToDevice));

    return CUDAPCAData(1, 1, patchSize, d_mean_copy);
#endif

    // Compute all possible binary products between the means of all the
    // random variables (i.e., 2^(patch size) possible products, ignoring
    // the commutative property). The resulting matrix contains in each
    // cell (i, j) the value E(Xi) * E(Xj). The BLAS function ger computes
    // A = axx^T + A. I use this function instead of spr because the
    // latter returns a packed matrix, and I want a full-fledged one.
    thrust::device_vector<data_t> d_prodmean(patchSize * patchSize, 0.f);
    alpha = 1.f;

    cublasCheck(cublasXger(handle, patchSize, patchSize,
                           &alpha,
                           d_mean.data().get(), 1,
                           d_mean.data().get(), 1,
                           d_prodmean.data().get(), patchSize));

//#define TEST_EIGEN_PRODMEAN
#ifdef TEST_EIGEN_PRODMEAN
    // Get values of mean product and return them
    data_t *d_prodmean_copy;

    cudaCheck(cudaMalloc((void**) &d_prodmean_copy,
                         patchSize * patchSize * sizeof(*d_prodmean_copy)));

    cudaCheck(cudaMemcpy(d_prodmean_copy, d_prodmean.data().get(),
                         patchSize * patchSize * sizeof(*d_prodmean_copy),
                         cudaMemcpyDeviceToDevice));

    return CUDAPCAData(1, patchSize, patchSize, d_prodmean_copy);
#endif

    // Compute the mean of all possible binary products of all the random
    // products (i.e., 2^(patch size) possible products, ignoring the
    // commutative property). The resulting matrix should contain in each
    // cell (i, j) the value E(Xi * Xj). This is done in several steps:
    //
    // 1.   Compute a matrix where each cell (i,j) is the sum of the
    //      products of the elements i and j of all patches. The resulting
    //      matrix contains in each cell (i, j) the value of sum(Xi * Xj).
    //      This is accomplished by multiplying the transposed patch
    //      matrix by itself (or the other way round in column-major
    //      format). BLAS gemm function computes C = aAB + bC.
    thrust::device_vector<data_t> d_sumprod(patchSize * patchSize);
    alpha = 1.f;
    beta = 0.f;

    cublasCheck(cublasXgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                            patchSize, patchSize, dataSize,
                            &alpha,
                            d_patches.data, patchSize,
                            d_patches.data, patchSize,
                            &beta,
                            d_sumprod.data().get(), patchSize));

//#define TEST_EIGEN_SUMPROD
#ifdef TEST_EIGEN_SUMPROD
    // Get values of product and return them
    data_t *d_sumprod_copy;

    cudaCheck(cudaMalloc((void**) &d_sumprod_copy,
                         patchSize * patchSize * sizeof(*d_sumprod_copy)));

    cudaCheck(cudaMemcpy(d_sumprod_copy, d_sumprod.data().get(),
                         patchSize * patchSize * sizeof(*d_sumprod_copy),
                         cudaMemcpyDeviceToDevice));

    return CUDAPCAData(1, patchSize, patchSize, d_sumprod_copy);
#endif

    // 2.   Divide the sum vector between the number of samples to
    //      get the mean of all products.
    thrust::device_vector<data_t> d_meanprod(d_sumprod);
    alpha = 1.f / dataSize;

    cublasCheck(cublasXscal(handle, patchSize * patchSize, &alpha,
                            d_meanprod.data().get(), 1));

//#define TEST_EIGEN_MEANPROD
#ifdef TEST_EIGEN_MEANPROD
    // Get values of mean product and return them
    data_t *d_meanprod_copy;

    cudaCheck(cudaMalloc((void**) &d_meanprod_copy,
                         patchSize * patchSize * sizeof(*d_meanprod_copy)));

    cudaCheck(cudaMemcpy(d_meanprod_copy, d_meanprod.data().get(),
                         patchSize * patchSize * sizeof(*d_meanprod_copy),
                         cudaMemcpyDeviceToDevice));

    return CUDAPCAData(1, patchSize, patchSize, d_meanprod_copy);
#endif

    // Compute the covariance matrix. This matrix contains all possible
    // covariance values for all the random variables. That is, cell
    // (i,j) contains the following value:
    //
    //              Cov(Xi, Xj) = E(Xi * Xj) - E(Xi) * E(Xj)
    //
    // This matrix is obtained from the results computed above. BLAS
    // famous function axpy computes y = ax + y.
    thrust::device_vector<data_t> d_cov(d_meanprod);
    alpha = -1.f;

    cublasCheck(cublasXaxpy(handle, patchSize * patchSize, &alpha,
                            d_prodmean.data().get(), 1,
                            d_cov.data().get(), 1));

//#define TEST_EIGEN_COV
#ifdef TEST_EIGEN_COV
    // Get values of covariance and return them
    data_t *d_cov_copy;

    cudaCheck(cudaMalloc((void**) &d_cov_copy,
            patchSize * patchSize * sizeof(*d_cov_copy)));

    cudaCheck(cudaMemcpy(d_cov_copy, d_cov.data().get(),
            patchSize * patchSize * sizeof(*d_cov_copy),
            cudaMemcpyDeviceToDevice));

    return CUDAPCAData(1, patchSize, patchSize, d_cov_copy);
#endif

    // Compute eigenvectors from the covariance matrix. The approach used
    // here is an iterative one. It is based on the code of Adams et al.
    // "Gaussian KD-Trees for Fast High-Dimensional Filtering" (2009).
    // Consider each row in the covariance matrix a first approximation
    // of each eigenvector. The code proceeds iteratively in several steps.
    // In each step, an orthonormal base is built from the previous
    // result (initially, the covariance matrix). If the result does not
    // converge, the covariance matrix is multiplied by this base, to
    // generate the next result to be used in the next iteration.
    //
    // I must say that the method present severe numerically inestabilities,
    // (and can result in computation of 1/0). This seems to be specially
    // aggravated by either the fact that computations are done in the GPU
    // or using CUBLAS (the CPU version seems to converge better, but it
    // does not mean. If the number of output dimensions (numPCADims)
    // is high, this method does not work. Using three or less dimensions
    // seems to be ok.
    thrust::device_vector<data_t> d_eigenvecs(d_cov);
    thrust::device_vector<data_t> d_eigenvecs_old(patchSize * patchSize, 0.f);

    saveGPUBuffer("d_eigenvecs_orig",
                  1, patchSize, patchSize,
                  d_eigenvecs);

    while(true) {
        // 1.   Find an orthogonal base for the current eigenvectors. This
        //      is achieved as a result of applying a Gram-Schmidt process
        //      that orthonormalizes the eigenvectors.
        for (int i = 0; i < numPCADims; i++) {
            // Make vector v_i independent of all previous ones
            for (int j = 0; j < i; j++) {
                // Compute dot = v_i * v_j, where * is the dot product
                data_t dot = 0.f;
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

            // Normalize vector
            // Compute norm = norm(v_i)
            data_t norm = 0.f;
            cublasCheck(cublasXnrm2(handle, patchSize,
                                    d_eigenvecs.data().get() + i*patchSize, 1,
                                    &norm));

            // Compute v_i = v_i / norm, and ensure the first component
            // of each eigenvector is positive
            norm = 1.f / norm;
            if (d_eigenvecs[i*patchSize] < 0) norm = -norm;

            cublasCheck(cublasXscal(handle, patchSize, &norm,
                                    d_eigenvecs.data().get() + i*patchSize, 1))
        }

        // 2. Check convergence
        // Compute d_diff = d_eigen - d_old
        thrust::device_vector<data_t> d_diff(d_eigenvecs);

        alpha = -1.0f;
        cublasCheck(cublasXaxpy(handle, patchSize * patchSize,
                                &alpha,
                                d_eigenvecs_old.data().get(), 1,
                                d_diff.data().get(), 1));

        // Accumulate squared norm of the difference vectors
        data_t dist = 0.f;
        for (int i = 0; i < numPCADims; i++) {
            data_t dot = 0.f;
            cublasCheck(cublasXdot(handle, patchSize,
                                   d_diff.data().get() + i*patchSize, 1,
                                   d_diff.data().get() + i*patchSize, 1,
                                   &dot));

            dist += dot;
        }

        // Break loop if solution has converged
        printf("Distance to convergence = %12.6f\n", dist);
        if (dist < 0.001f) break;

        // Solution has not converged, so we keep iterating. Multiply
        // the covariance matrix by current eigenvectors
        cudaCheck(cudaMemset(d_eigenvecs_old.data().get(),
                             0, patchSize * patchSize * sizeof(data_t)));
        alpha = 1.f;
        beta = 1.f;

        cublasCheck(cublasXgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                patchSize, patchSize, patchSize,
                                &alpha,
                                d_cov.data().get(), patchSize,
                                d_eigenvecs.data().get(), patchSize,
                                &beta,
                                d_eigenvecs_old.data().get(), patchSize));

        // Swap old and current eigenvectors
        d_eigenvecs_old.swap(d_eigenvecs);
    }

    // Clean and return
    cublasCheck(cublasDestroy(handle));
    return CUDAPCAData(1, numPCADims, patchSize, d_eigenvecs.data().get());
}


CUDAPCA::CUDAPCAData
CUDAPCA::projectPatches(
        const CUDAPCAPatches &d_patches,
        const CUDAPCAData &d_eigenvecs)
{
    // Create CUBLAS handle
    cublasHandle_t handle;
    cublasCheck(cublasCreate(&handle));

    const int dataSize = d_patches.depth * d_patches.height * d_patches.width;
    const int patchDiam = (2 * d_patches.patchRadius + 1);
    const int patchSize = patchDiam * patchDiam * patchDiam;
    const int numPCADims = d_eigenvecs.height;

    // Multiply all patches by the current vectors. CUBLAS function gemm
    // computes C = aAB + bC. Notice that patches and eigenvectors are
    // stored in row-wise fashion. From CUBLAS point of view, this means
    // that each column is a patch or an eigenvector, respectively. As
    // matrix multiplication multiplies each row in A by each column B,
    // the matrix with the patches must be transposed.
    thrust::device_vector<data_t> d_projPatches(dataSize * numPCADims, 0.f);
    data_t alpha = 1.f;
    data_t beta = 1.f;

    cublasCheck(cublasXgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            dataSize, numPCADims, patchSize,
                            &alpha,
                            d_patches.data, patchSize,
                            d_eigenvecs.data, patchSize,
                            &beta,
                            d_projPatches.data().get(), dataSize));

    // CUBLAS uses a column-major format. The resulting matrix from the
    // previous operation needs to be transposed in order to get the matrix
    // stored in a row-major format (a la C). CUBLAS function geam computes
    // C = aA + bB, and B is ignored if b is zero.
    thrust::device_vector<data_t> d_projPatchesTrans(dataSize * numPCADims);
    alpha = 1.f;
    beta = 0.f;

    cublasCheck(cublasXgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            numPCADims, dataSize,
                            &alpha,
                            d_projPatches.data().get(), dataSize,
                            &beta,
                            d_projPatches.data().get(), dataSize,
                            d_projPatchesTrans.data().get(), numPCADims));

    // Clean and return
    cublasCheck(cublasDestroy(handle));
    return CUDAPCAData(1, dataSize, numPCADims,
                       d_projPatchesTrans.data().get());
}
