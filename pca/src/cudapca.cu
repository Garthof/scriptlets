#include "cudapca.h"

#include <cstdio>
#include <cstdlib>

// Auxiliary functions

inline void
cudaCheckReturn(const cudaError_t stat) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "Error %s at line %d in file %s\n",
                cudaGetErrorString(stat), __LINE__, __FILE__);
        exit(EXIT_FAILURE);
    }
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
    cudaCheckReturn(cudaFree(cData));
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
    cudaCheckReturn(cudaMalloc((void**) &d_data,
                               dataSize * sizeof(*d_data)));

    cudaCheckReturn(cudaMemcpy(d_data, h_data,
                               dataSize * sizeof(*d_data),
                               cudaMemcpyHostToDevice));

    // Generate CUDAPCAData object and return
    CUDAPCA::CUDAPCAData h_pcaData(depth, width, height, d_data);
    return h_pcaData;
}


std::auto_ptr<CUDAPCA::data_t>
CUDAPCA::downloadData(const CUDAPCA::CUDAPCAData &d_data)
{
    const int dataSize = d_data.depth * d_data.height * d_data.width;
    data_t *h_data;

    // Allocate space in host and copy data from GPU memory
    cudaCheckReturn(cudaMallocHost((void **) &h_data,
                                   dataSize * sizeof(*h_data)));

    cudaCheckReturn(cudaMemcpy(h_data, d_data.data,
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
    const int patchDiam = (2 * patchRadius + 1);
    const int patchSize = patchDiam * patchDiam * patchDiam;
    const int dataSize = d_data.depth * d_data.height * d_data.width;

    data_t *d_patchData;
    cudaCheckReturn(cudaMalloc((void **) &d_patchData,
                               patchSize * dataSize * sizeof(*d_patchData)));

    // Launch kernel and wait to finish
    const dim3 dimBlk(512);
    const dim3 dimGrd(ceilDiv(dataSize, dimBlk.x));

    kernGeneratePatches<<<dimGrd, dimBlk>>>(
            d_patchData, d_data.data,
            d_data.depth, d_data.width, d_data.height,
            patchRadius);

    cudaCheckReturn(cudaThreadSynchronize());
    cudaCheckReturn(cudaGetLastError());

    return CUDAPCA::CUDAPCAPatches(
            d_data.depth, d_data.width, d_data.height,
            patchRadius, d_patchData);
}


std::auto_ptr<CUDAPCA::data_t>
CUDAPCA::downloadPatches(const CUDAPCA::CUDAPCAPatches &d_patches)
{
    const int patchDiam = (2 * d_patches.patchRadius + 1);
    const int patchSize = patchDiam * patchDiam * patchDiam;
    const int patchesSize = d_patches.depth * d_patches.height
                            * d_patches.width * patchSize;
    data_t *h_patches;

    // Allocate space in host and copy data from GPU memory
    cudaCheckReturn(cudaMallocHost((void **) &h_patches,
                                   patchesSize * sizeof(*h_patches)));

    cudaCheckReturn(cudaMemcpy(h_patches, d_patches.data,
                               patchesSize * sizeof(*h_patches),
                               cudaMemcpyDeviceToHost));

    return std::auto_ptr<data_t>(h_patches);
}
