// Constructs a 3D texture where each element is a float4. By default,
// texture coordinates are not normalized, no interpolation is done to
// fetch the values (filter mode is point), and the addressing mode is
// clamp.
texture<float, cudaTextureType1D> t_dataSM;
texture<float4, cudaTextureType1D> t_dataSM4;


__device__ __forceinline__ float
kernTexFilterHorSMOperation1(
        const float &val0, const float &val1, const float a)
{
    float res = val0;
    res += a * (val1 - val0);
    return res;
}


__device__ __forceinline__ float4
kernTexFilterHorSMOperation4(
        const float4 &val0, const float4 &val1, const float a)
{
    float4 res = val0;

    res.x += a * (val1.x - val0.x);
    res.y += a * (val1.y - val0.y);
    res.z += a * (val1.z - val0.z);
    res.w += a * (val1.w - val0.w);

    return res;
}


//template<int nn>
//__global__ void
//kernTexFilterHorSM(
//        float *const g_resData,
//        const int depth, const int height, const int width,
//        const float a)
//{
//    const int j = threadIdx.x + blockIdx.x * blockDim.x;
//    const int k = threadIdx.y + blockIdx.y * blockDim.y;
//
//    if (j < height && k < depth) {
//        // Note on coordinates: using point filter mode is equivalent to a
//        // nearest point sampling. By CUDA's specification, this means that
//        // tex1D(x) == T[(int) floor(x)]. To achieve a correct conversion
//        // during flooring, it is convenient to use tex1D(x+.5f) in the code.
//        float prevVal[nn];
//        const int base = (j * width + k * width * height) * nn;
////        printf("i = 0: ");
//
//        for (int n = 0; n < nn; n++) {
//            g_resData[base+n] = prevVal[n] = tex1Dfetch(t_data, base+n);
////            printf("%f, ", prevVal[n]);
//        }
////        printf("\n");
//
//        for (int i = 1; i < width; i++) {
////            printf("i = %d: ", i);
//            for (int n = 0; n < nn; n++) {
//                const float curVal = tex1Dfetch(t_data, base+(i*nn+n));
//                const float newVal = kernTexFilterHorOperation1(curVal, prevVal[n], a);
//                g_resData[base+(i*nn+n)] = prevVal[n] = newVal;
////                printf("%f, ", curVal);
//            }
////            printf("\n");
//        }
//    }
//}


// The variable pointing to dinamically allocated shared memory must be
// declared as an extern array. Otherwise, expect undefined behavior.
extern __shared__ char s_mem[];


__global__ void
//__launch_bounds__(8*8)
kernTexFilterHorSM(
        float4 *const g_resData,
        const int g_depth, const int g_height, const int g_width,
        const float a)
{
    const int g_k = threadIdx.y + blockIdx.y * blockDim.y;
    const int g_j = threadIdx.x + blockIdx.x * blockDim.x;

    if (g_j < g_height && g_k < g_depth) {
        float4 *const s_data = (float4 *) s_mem;

        const int s_depth = blockDim.y;
        const int s_height = blockDim.x;
        const int s_width = g_width;

        const int s_k = g_k % s_depth;
        const int s_j = g_j % s_height;

        const int g_base = g_j * g_width + g_k * g_width * g_height;
        const int s_base = s_j * s_width + s_k * s_width * s_height;

        // Causal filter
        float4 prevVal;
        s_data[s_base+0] = prevVal = tex1Dfetch(t_dataSM4, g_base+0);
//        printf("i = 0: %f, %f, %f, %f\n", prevVal.x, prevVal.y, prevVal.z, prevVal.w);

        for (int i = 1; i < g_width; i++) {
            const float4 curVal = tex1Dfetch(t_dataSM4, g_base+i);
            const float4 newVal = kernTexFilterHorSMOperation4(curVal, prevVal, a);
            s_data[s_base+i] = prevVal = newVal;
//            printf("i = %d: %f, %f, %f, %f\n", i, curVal.x, curVal.y, curVal.z, curVal.w);
        }

        // Anticausal filter
        for (int i = g_width-2; i >= 0; i--) {
            const float4 curVal = s_data[s_base+i];
            const float4 newVal = kernTexFilterHorSMOperation4(curVal, prevVal, a);
            g_resData[g_base+i] = prevVal = newVal;
        }
    }
}


//template<int nn>
//void
//computeResultFromTexture(
//        float *const h_resData,
//        const float *const h_origData,
//        const int depth, const int height, const int width,
//        const float a)
//{
//    const int dataSize = depth * height * width * nn;
//
//    // Copy original data in GPU
//    float *d_origData;
//    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_origData,
//                                 sizeof(*d_origData) * dataSize));
//
//    CUDA_CHECK_RETURN(cudaMemcpy(d_origData, h_origData,
//                                 sizeof(*d_origData) * dataSize,
//                                 cudaMemcpyHostToDevice));
//
//    // Generate buffer to store result in GPU
//    float *d_resData;
//    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_resData,
//                                 sizeof(*d_resData) * dataSize));
//
//    // Compute result
//    const dim3 blkSize(8, 8);
//    const dim3 grdSize(ceil_div(height, (int) blkSize.x),
//                       ceil_div(depth, (int) blkSize.y));
//
//    for (int i = 0; i < NUM_ITERS; i++) {
//        CUDA_CHECK_RETURN(cudaBindTexture(0, t_dataSM, d_origData,
//                                          sizeof(*d_origData) * dataSize));
//
//        kernTexFilterHor1<nn><<<grdSize, blkSize>>>
//                             (d_resData, depth, height, width, a);
//
//        CUDA_CHECK_RETURN(cudaBindTexture(0, t_dataSM, d_resData,
//                                          sizeof(*d_resData) * dataSize));
//
//        kernTexFilterHor2<nn><<<grdSize, blkSize>>>
//                             (d_origData, depth, height, width, a);
//    }
//
//    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
//    CUDA_CHECK_RETURN(cudaGetLastError());
//
//    // Unbind texture
//    CUDA_CHECK_RETURN(cudaUnbindTexture(t_dataSM));
//
//    // Copy result into CPU memory
//    CUDA_CHECK_RETURN(cudaMemcpy(h_resData, d_origData,
//                                 sizeof(*d_origData) * dataSize,
//                                 cudaMemcpyDeviceToHost));
//
//    CUDA_CHECK_RETURN(cudaFree(d_origData));
//    CUDA_CHECK_RETURN(cudaFree(d_resData));
//}



void
computeResultFromTextureSM(
        float4 *const h_resData,
        const float4 *const h_origData,
        const int depth, const int height, const int width, const int nn,
        const float a)
{
    const int dataSize = depth * height * width * nn;

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

    // Bind texture
    CUDA_CHECK_RETURN(cudaBindTexture(0, t_dataSM4, d_origData,
                                      sizeof(*d_origData) * dataSize));

    // Compute result
    const dim3 blkSize(4, 4);
    const dim3 grdSize(ceil_div(height, (int) blkSize.x),
                       ceil_div(depth, (int) blkSize.y));

    for (int i = 0; i < NUM_ITERS; i++) {
        kernTexFilterHorSM<<<grdSize, blkSize,
                             width * blkSize.x * blkSize.y * sizeof(float4)>>>
                          (d_resData, depth, height, width, a);

        CUDA_CHECK_RETURN(cudaThreadSynchronize());
        CUDA_CHECK_RETURN(cudaGetLastError());
    }

    // Unbind texture
    CUDA_CHECK_RETURN(cudaUnbindTexture(t_dataSM4));

    // Copy result into CPU memory
    CUDA_CHECK_RETURN(cudaMemcpy(h_resData, d_resData,
                                 sizeof(*d_resData) * dataSize,
                                 cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(d_origData));
    CUDA_CHECK_RETURN(cudaFree(d_resData));
}
