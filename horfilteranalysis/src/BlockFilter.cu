#include "BlockFilter.h"

#include <cstdio>

#include "constants.h"
#include "CUDAUtils.h"

#define BLK_DEPTH 1
#define BLK_HEIGHT 32
#define BLK_WIDTH BLK_HEIGHT

texture<float, cudaTextureType1D> t_dataB;
texture<float4, cudaTextureType1D> t_dataB4;

__constant__ float4 c_AFP[BLK_WIDTH];
__constant__ float4 c_AbF;
__constant__ float4 c_AbR;
__constant__ float4 c_HARB_AFP;


#ifdef __CUDA_ARCH__
__host__ __device__ __forceinline__
#else
inline
#endif
float4
filterHorOperation4(
        const float4 &val0, const float4 &val1, const float a)
{
    float4 res = val0;

    res.x += a * (val1.x - val0.x);
    res.y += a * (val1.y - val0.y);
    res.z += a * (val1.z - val0.z);
    res.w += a * (val1.w - val0.w);

    return res;
}


#ifdef __CUDA_ARCH__
__host__ __device__ __forceinline__
#else
inline
#endif
float4
multiplyAdd(const float4 &add0, const float4 &fact0, const float4 &fact1)
{
    float4 res = add0;

    res.x += fact0.x * fact1.x;
    res.y += fact0.y * fact1.y;
    res.z += fact0.z * fact1.z;
    res.w += fact0.w * fact1.w;

    return res;
}


__global__ void
kernComputePYbarAndEzhat(
        float4 *const g_pybar, float4 *const g_ezhat,
        const int volDepth, const int volHeight, const int volWidth,
        const float a)
{
    const int tj = threadIdx.x;
    const int tk = threadIdx.y;
    const int bi = blockIdx.x;
    const int bj = blockIdx.y;
    const int bk = blockIdx.z;

    const int base_i = BLK_WIDTH * bi;
    const int base_j = BLK_HEIGHT * bj;
    const int base_k = BLK_DEPTH * bk;

    if (base_j + tj >= volHeight) return;

    // Compute Ybar = F(0, B(X)) in s_data
    __shared__ float4 s_data[BLK_DEPTH][BLK_HEIGHT][BLK_WIDTH];
    float4 prevVal;

    const int base =
            base_i
            + (base_j + tj) * volWidth
            + (base_k + tk) * volWidth * volHeight;

    if (bi != 0) {
        prevVal = make_float4(0.f, 0.f, 0.f, 0.f);
        const float4 curVal = tex1Dfetch(t_dataB4, base+0);
        const float4 newVal = filterHorOperation4(curVal, prevVal, a);
        s_data[tk][tj][0] = prevVal = newVal;
    } else {
        s_data[tk][tj][0] = prevVal = tex1Dfetch(t_dataB4, base+0);
    }

    if (bi < gridDim.x-1) {
#pragma unroll
        for (int i = 1; i < BLK_WIDTH; i++) {
            const float4 curVal = tex1Dfetch(t_dataB4, base+i);
            const float4 newVal = filterHorOperation4(curVal, prevVal, a);
            s_data[tk][tj][i] = prevVal = newVal;
        }
    } else {
        for (int i = 1; i < volWidth%BLK_WIDTH; i++) {
            const float4 curVal = tex1Dfetch(t_dataB4, base+i);
            const float4 newVal = filterHorOperation4(curVal, prevVal, a);
            s_data[tk][tj][i] = prevVal = newVal;
        }
    }

    // Store PYbar as T(Ybar)
//    const int appDepth = volDepth;      // Appendix depth
    const int appHeight = volHeight;    // Appendix height
    const int appWidth = gridDim.x;     // Appendix width

    const int appPos =
            bi
            + (base_j + tj) * appWidth
            + (base_k + tk) * appWidth * appHeight;
    g_pybar[appPos] = prevVal;

    // Compute Zhat = R(Ybar) from s_data
    if (bi < appWidth-1) {
        prevVal = make_float4(0.f, 0.f, 0.f, 0.f);

#pragma unroll
        for (int i = BLK_WIDTH - 1; i >= 0; i--) {
            const float4 curVal = s_data[tk][tj][i];
            const float4 newVal = filterHorOperation4(curVal, prevVal, a);
            prevVal = newVal;
        }
    } else {
        for (int i = (volWidth%BLK_WIDTH) - 2; i >= 0; i--) {
            const float4 curVal = s_data[tk][tj][i];
            const float4 newVal = filterHorOperation4(curVal, prevVal, a);
            prevVal = newVal;
        }
    }

    // Store EZhat as H(Zhat)
    g_ezhat[appPos] = prevVal;
}


__global__ void
kernComputePYandEZ(
        float4 *const g_pybar, float4 *const g_ezhat,
        const int volDepth, const int volHeight, const int volWidth)
{
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int k = threadIdx.y + blockIdx.y * blockDim.y;

    const int appDepth = volDepth;                          // Appendix depth
    const int appHeight = volHeight;                        // Appendix height
    const int appWidth = ceil_div(volWidth, BLK_WIDTH);     // Appendix width

    if (j < appHeight && k < appDepth) {
        const int base = j * appWidth + k * appWidth * appHeight;
        float4 prevValY = g_pybar[base+0];

        // Compute Pm(Y) = Pm(Ybar) + AbF * Pm-1(Y)
        for (int i = 1; i < appWidth; i++) {
            const float4 curValYbar = g_pybar[base+i];
            const float4 newValY = multiplyAdd(curValYbar, prevValY, c_AbF);
            g_pybar[base+i] = prevValY = newValY;
        }

        // Compute Em(Z) = AbR * Em+1(Z) + (H(ARB) * AFP) * Pm-1(Y) * Em(Zhat)
        float4 prevValZ;
        float4 curValZhat = g_ezhat[base+appWidth-1];
        float4 newValZ = multiplyAdd(curValZhat, c_HARB_AFP, prevValY);
        g_ezhat[base+appWidth-1] = prevValZ = newValZ;

        for (int i = appWidth-2; i >= 1; i--) {
            curValZhat = g_ezhat[base+i];
            newValZ = multiplyAdd(curValZhat, c_AbR, prevValZ);
            newValZ = multiplyAdd(newValZ, c_HARB_AFP, g_pybar[base+i-1]);
            g_ezhat[base+i] = prevValZ = newValZ;
        }

        curValZhat = g_ezhat[base+0];
        newValZ = multiplyAdd(curValZhat, c_AbR, prevValZ);
        g_ezhat[base+0] = prevValZ = newValZ;
    }
}


__global__ void
kernComputeYandZ(
        float4 *const g_data,
        const float4 *const g_py, const float4 *const g_ez,
        const int volDepth, const int volHeight, const int volWidth,
        const float a)
{
    const int tj = threadIdx.x;
    const int tk = threadIdx.y;
    const int bi = blockIdx.x;
    const int bj = blockIdx.y;
    const int bk = blockIdx.z;

    const int base_i = BLK_WIDTH * bi;
    const int base_j = BLK_HEIGHT * bj;
    const int base_k = BLK_DEPTH * bk;

    if (base_j + tj >= volHeight) return;

    __shared__ float4 s_data[BLK_DEPTH][BLK_HEIGHT][BLK_WIDTH];
//    const int appDepth = volDepth;      // Appendix depth
    const int appHeight = volHeight;    // Appendix height
    const int appWidth = gridDim.x;     // Appendix width

    const int base =
            base_i
            + (base_j + tj) * volWidth
            + (base_k + tk) * volWidth * volHeight;

    const int appPos =
            bi
            + (base_j + tj) * appWidth
            + (base_k + tk) * appWidth * appHeight;

    // Compute Bm(Y) in s_data from Pm-1(Y)
    float4 prevVal;

    if (bi > 0) {
        prevVal = g_py[appPos-1];
        const float4 curVal = tex1Dfetch(t_dataB4, base+0);
        const float4 newVal = filterHorOperation4(curVal, prevVal, a);
        s_data[tk][tj][0] = prevVal = newVal;
    } else {
        const float4 curVal = tex1Dfetch(t_dataB4, base+0);
        s_data[tk][tj][0] = prevVal = curVal;
    }

    if (bi < appWidth-1) {
#pragma unroll
        for (int i = 1; i < BLK_WIDTH; i++) {
            const float4 curVal = tex1Dfetch(t_dataB4, base+i);
            const float4 newVal = filterHorOperation4(curVal, prevVal, a);
            s_data[tk][tj][i] = prevVal = newVal;
        }
    } else {
        for (int i = 1; i < volWidth%BLK_WIDTH; i++) {
            const float4 curVal = tex1Dfetch(t_dataB4, base+i);
            const float4 newVal = filterHorOperation4(curVal, prevVal, a);
            s_data[tk][tj][i] = prevVal = newVal;
        }
    }

    // Compute Bm(Z) from s_data and Em+1(Z)
    if (bi < appWidth-1) {
        prevVal = g_ez[appPos+1];

#pragma unroll
        for (int i = BLK_WIDTH - 1; i >= 0; i--) {
            const float4 curVal = s_data[tk][tj][i];
            const float4 newVal = filterHorOperation4(curVal, prevVal, a);
            g_data[base+i] = prevVal = newVal;
        }
    } else {
        float4 curVal = s_data[tk][tj][(volWidth%BLK_WIDTH)-1];
        g_data[base+(volWidth%BLK_WIDTH)-1] = prevVal = curVal;

        for (int i = (volWidth%BLK_WIDTH) - 2; i >= 0; i--) {
            curVal = s_data[tk][tj][i];
            const float4 newVal = filterHorOperation4(curVal, prevVal, a);
            g_data[base+i] = prevVal = newVal;
        }
    }
}


void
computeConstants(const float a)
{
    // Compute AFP = F(Ir, 0), where Ir is an rxr unit matrix
    // and 0 is an rxb zero matrix. In this case, r=1, and b=BLK_WIDTH.
    float4 h_AFP[1][BLK_WIDTH];
    float4 prevVal = make_float4(1.f, 1.f, 1.f, 1.f);

    for (int i = 0; i < BLK_WIDTH; i++) {
        const float4 curVal = make_float4(0.f, 0.f, 0.f, 0.f);
        h_AFP[0][i] = prevVal = filterHorOperation4(curVal, prevVal, a);
    }

    // Compute ARE = R(0, Ir)
    float4 h_ARE[1][BLK_WIDTH];
    prevVal = make_float4(1.f, 1.f, 1.f, 1.f);

    for (int i = BLK_WIDTH-1; i >= 0; i--) {
        const float4 curVal = make_float4(0.f, 0.f, 0.f, 0.f);
        h_ARE[0][i] = prevVal = filterHorOperation4(curVal, prevVal, a);
    }

//    // Compute AFB = F(0, Ib), where Ib is a bxb unit matrix and 0 is
//    // a bxr zero matrix. In this case, r=1, and b=BLK_WIDTH=BLK_HEIGHT.
//    float4 h_AFB[BLK_HEIGHT][BLK_WIDTH];
//
//    for (int j = 0; j < BLK_HEIGHT; j++) {
//        prevVal = make_float4(0.f, 0.f, 0.f, 0.f);
//        for (int i = 0; i < BLK_WIDTH; i++) {
//            const float4 curVal = (i != j) ? make_float4(0.f, 0.f, 0.f, 0.f)
//                                           : make_float4(1.f, 1.f, 1.f, 1.f);
//            h_AFB[j][i] = prevVal = filterHorOperation4(curVal, prevVal, a);
//        }
//    }

    // Compute ARB = R(Ib, 0)
    float4 h_ARB[BLK_HEIGHT][BLK_WIDTH];

    for (int j = 0; j < BLK_HEIGHT; j++) {
        prevVal = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int i = BLK_WIDTH; i >= 0; i--) {
            const float4 curVal = (i != j) ? make_float4(0.f, 0.f, 0.f, 0.f)
                                           : make_float4(1.f, 1.f, 1.f, 1.f);
            h_ARB[j][i] = prevVal = filterHorOperation4(curVal, prevVal, a);
        }
    }

    // Compute AbF = T(AFP) and AbR = H(ARE)
    const float4 h_AbF = h_AFP[0][BLK_WIDTH-1];
    const float4 h_AbR = h_ARE[0][0];

    // Compute H(ARB) * AFP
    float4 h_HARB_AFP = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int i = 0; i < BLK_HEIGHT; i++) {
        h_HARB_AFP = multiplyAdd(h_HARB_AFP, h_AFP[0][i], h_ARB[i][0]);
    }

    // Load symbols into constant memory
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_AFP, &h_AFP[0], sizeof(h_AFP),
                                         0, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_AbF, &h_AbF, sizeof(h_AbF),
                                         0, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_AbR, &h_AbR, sizeof(h_AbR),
                                         0, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_HARB_AFP, &h_HARB_AFP,
                                         sizeof(h_HARB_AFP),
                                         0, cudaMemcpyHostToDevice));
}


void
computePYbarAndEZhat(
        float4 *const d_pybar, float4 *const d_ezhat,
        const int depth, const int height, const int width,
        const float a)
{
    const dim3 blkSize(BLK_HEIGHT, BLK_DEPTH);
    const dim3 grdSize(ceil_div(width, BLK_WIDTH),
                       ceil_div(height, BLK_HEIGHT),
                       ceil_div(depth, BLK_DEPTH));

    kernComputePYbarAndEzhat<<<grdSize, blkSize>>>
                            (d_pybar, d_ezhat, depth, height, width, a);

    CUDA_CHECK_RETURN(cudaThreadSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());
}


void
computePYandEZ(
        float4 *const d_pybar, float4 *const d_ezhat,
        const int depth, const int height, const int width)
{
    const dim3 blkSize(BLK_HEIGHT, BLK_DEPTH);
    const dim3 grdSize(ceil_div(height, BLK_HEIGHT),
                       ceil_div(depth, BLK_DEPTH));

    kernComputePYandEZ<<<grdSize, blkSize>>>
                      (d_pybar, d_ezhat, depth, height, width);

    CUDA_CHECK_RETURN(cudaThreadSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());
}


void
computeYandZ(
        float4 *const d_data,
        const float4 *const d_py, const float4 *const d_ez,
        const int depth, const int height, const int width,
        const float a)
{
    const dim3 blkSize(BLK_HEIGHT, BLK_DEPTH);
    const dim3 grdSize(ceil_div(width, BLK_WIDTH),
                       ceil_div(height, BLK_HEIGHT),
                       ceil_div(depth, BLK_DEPTH));

    kernComputeYandZ<<<grdSize, blkSize>>>
                    (d_data, d_py, d_ez, depth, height, width, a);

    CUDA_CHECK_RETURN(cudaThreadSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());
}


void
computeBlockResultFromTexture(
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

    // Generate buffer to store result in GPU
    float4 *d_resData;
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_resData,
                                 sizeof(*d_resData) * dataSize));

    // Generate buffer to store P(Ybar) and E(Zhat)
    const int numBlocksDepth = ceil_div(depth, BLK_DEPTH);;
    const int numBlocksHeight = ceil_div(height, BLK_HEIGHT);
    const int numBlocksWidth = ceil_div(width, BLK_WIDTH);
    const int numBlocksTotal = numBlocksDepth * numBlocksHeight * numBlocksWidth;

    float4 *d_pybar;
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_pybar,
            numBlocksTotal * BLK_HEIGHT * BLK_DEPTH * sizeof(*d_pybar)));

    float4 *d_ezhat;
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_ezhat,
            numBlocksTotal * BLK_HEIGHT * BLK_DEPTH * sizeof(*d_pybar)));

    CUDA_CHECK_RETURN(cudaBindTexture(0, t_dataB4, d_origData,
                                      sizeof(*d_origData) * dataSize));

    for (int i = 0; i < NUM_ITERS; i++) {
        computePYbarAndEZhat(d_pybar, d_ezhat,
                             depth, height, width, a);

        computePYandEZ(d_pybar, d_ezhat, depth, height, width);

        computeYandZ(d_resData, d_pybar, d_ezhat,
                     depth, height, width, a);
    }

    // Unbind texture
    CUDA_CHECK_RETURN(cudaUnbindTexture(t_dataB4));

    // Copy result into CPU memory
    CUDA_CHECK_RETURN(cudaMemcpy(h_resData, d_resData,
                                 sizeof(*d_origData) * dataSize,
                                 cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(d_origData));
    CUDA_CHECK_RETURN(cudaFree(d_resData));
}
