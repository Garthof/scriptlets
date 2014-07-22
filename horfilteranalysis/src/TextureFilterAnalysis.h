texture<float, cudaTextureType1D> t_data3D;
texture<float4, cudaTextureType1D> t_data4;


__device__ __forceinline__ float
kernTexFilterHorOperation1(
        const float &val0, const float &val1, const float a)
{
    float res = val0;
    res += a * (val1 - val0);
    return res;
}


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


template<int nn>
__global__ void
kernTexFilterHor1(
        float *const g_resData,
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
        float prevVal[nn];
        const int base = (j * width + k * width * height) * nn;
//        printf("i = 0: ");

        for (int n = 0; n < nn; n++) {
            g_resData[base+n] = prevVal[n] = tex1Dfetch(t_data3D, base+n);
//            printf("%f, ", prevVal[n]);
        }
//        printf("\n");

        for (int i = 1; i < width; i++) {
//            printf("i = %d: ", i);
            for (int n = 0; n < nn; n++) {
                const float curVal = tex1Dfetch(t_data3D, base+(i*nn+n));
                const float newVal = kernTexFilterHorOperation1(curVal, prevVal[n], a);
                g_resData[base+(i*nn+n)] = prevVal[n] = newVal;
//                printf("%f, ", curVal);
            }
//            printf("\n");
        }
    }
}


template<int nn>
__global__ void
kernTexFilterHor2(
        float *const g_resData,
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
        float prevVal[nn];
        const int base = (j * width + k * width * height) * nn;
//        printf("i = 0: ");

        for (int n = 0; n < nn; n++) {
            g_resData[base+((width-1)*nn+n)] = prevVal[n] = tex1Dfetch(t_data3D, base+((width-1)*nn+n));
//            printf("%f, ", prevVal[n]);
        }
//        printf("\n");

        for (int i = width-2; i >= 0; i--) {
//            printf("i = %d: ", i);
            for (int n = 0; n < nn; n++) {
                const float curVal = tex1Dfetch(t_data3D, base+(i*nn+n));
                const float newVal = kernTexFilterHorOperation1(curVal, prevVal[n], a);
                g_resData[base+(i*nn+n)] = prevVal[n] = newVal;
//                printf("%f, ", curVal);
            }
//            printf("\n");
        }
    }
}


__global__ void
kernTexFilterHor1(
        float4 *const g_resData,
        const int depth, const int height, const int width, const int nn,
        const float a)
{
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int k = threadIdx.y + blockIdx.y * blockDim.y;

    if (j < height && k < depth) {
        float4 prevVal;
        const int base = j * width + k * width * height;
        g_resData[base+0] = prevVal = tex1Dfetch(t_data4, base+0);
//        printf("i = 0: %f, %f, %f, %f\n", prevVal.x, prevVal.y, prevVal.z, prevVal.w);
        for (int i = 1; i < width; i++) {
            const float4 curVal = tex1Dfetch(t_data4, base+i);
            const float4 newVal = kernTexFilterHorOperation4(curVal, prevVal, a);
            g_resData[base+i] = prevVal = newVal;
//            printf("i = %d: %f, %f, %f, %f\n", i, curVal.x, curVal.y, curVal.z, curVal.w);
        }
    }
}


__global__ void
kernTexFilterHor2(
        float4 *const g_resData,
        const int depth, const int height, const int width, const int nn,
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
        g_resData[base+(width-1)] = prevVal = tex1Dfetch(t_data4, base+(width-1));

        for (int i = width-2; i >= 0; i--) {
            const float4 curVal = tex1Dfetch(t_data4, base+i);
            const float4 newVal = kernTexFilterHorOperation4(curVal, prevVal, a);
            g_resData[base+i] = prevVal = newVal;
        }
    }
}


__global__ void
kernCtrlTexFilterHor1(
        float4 *const g_resData, const float4 *const g_origData,
        const int depth, const int height, const int width, const int nn,
        const float a)
{
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int k = threadIdx.y + blockIdx.y * blockDim.y;

    if (j < height && k < depth) {
        float4 prevVal;
        const int base = j * width + k * width * height;
        g_resData[base+0] = prevVal = g_origData[base+0];

        for (int i = 1; i < width; i++) {
            const float4 curVal = g_origData[base+i];
            const float4 newVal = kernTexFilterHorOperation4(curVal, prevVal, a);

            g_resData[base+i] = prevVal = newVal;
        }
    }
}


__global__ void
kernCtrlTexFilterHor2(
        float4 *const g_resData, const float4 *const g_origData,
        const int depth, const int height, const int width, const int nn,
        const float a)
{
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int k = threadIdx.y + blockIdx.y * blockDim.y;

    if (j < height && k < depth) {
        float4 prevVal;
        const int base = j * width + k * width * height;
        g_resData[base+(width-1)] = prevVal = g_origData[base+(width-1)];

        for (int i = width-2; i >= 0; i--) {
            const float4 curVal = g_origData[base+i];
            const float4 newVal = kernTexFilterHorOperation4(curVal, prevVal, a);

            g_resData[base+i] = prevVal = newVal;
        }
    }
}


template<int nn>
void
computeResultFromTexture(
        float *const h_resData,
        const float *const h_origData,
        const int depth, const int height, const int width,
        const float a)
{
    const int dataSize = depth * height * width * nn;

    // Copy original data in GPU
    float *d_origData;
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_origData,
                                 sizeof(*d_origData) * dataSize));

    CUDA_CHECK_RETURN(cudaMemcpy(d_origData, h_origData,
                                 sizeof(*d_origData) * dataSize,
                                 cudaMemcpyHostToDevice));

    // Generate buffer to store result in GPU
    float *d_resData;
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_resData,
                                 sizeof(*d_resData) * dataSize));

    // Compute result
    const dim3 blkSize(8, 8);
    const dim3 grdSize(ceil_div(height, (int) blkSize.x),
                       ceil_div(depth, (int) blkSize.y));

    for (int i = 0; i < NUM_ITERS; i++) {
        CUDA_CHECK_RETURN(cudaBindTexture(0, t_data3D, d_origData,
                                          sizeof(*d_origData) * dataSize));

        kernTexFilterHor1<nn><<<grdSize, blkSize>>>
                             (d_resData, depth, height, width, a);

        CUDA_CHECK_RETURN(cudaBindTexture(0, t_data3D, d_resData,
                                          sizeof(*d_resData) * dataSize));

        kernTexFilterHor2<nn><<<grdSize, blkSize>>>
                             (d_origData, depth, height, width, a);
    }

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());

    // Unbind texture
    CUDA_CHECK_RETURN(cudaUnbindTexture(t_data3D));

    // Copy result into CPU memory
    CUDA_CHECK_RETURN(cudaMemcpy(h_resData, d_origData,
                                 sizeof(*d_origData) * dataSize,
                                 cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(d_origData));
    CUDA_CHECK_RETURN(cudaFree(d_resData));
}



void
computeResultFromTexture(
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

    // Compute result
    const dim3 blkSize(8, 8);
    const dim3 grdSize(ceil_div(height, (int) blkSize.x),
                       ceil_div(depth, (int) blkSize.y));

    for (int i = 0; i < NUM_ITERS; i++) {
        CUDA_CHECK_RETURN(cudaBindTexture(0, t_data4, d_origData,
                                          sizeof(*d_origData) * dataSize));

        kernTexFilterHor1<<<grdSize, blkSize>>>(d_resData, depth, height, width,
                                                nn, a);
//        kernCtrlTexFilterHor1<<<grdSize, blkSize>>>(d_resData, d_origData,
//                                                    depth, height, width, nn, a);

        CUDA_CHECK_RETURN(cudaBindTexture(0, t_data4, d_resData,
                                          sizeof(*d_resData) * dataSize));

        kernTexFilterHor2<<<grdSize, blkSize>>>(d_origData, depth, height, width,
                                                nn, a);
//        kernCtrlTexFilterHor2<<<grdSize, blkSize>>>(d_origData, d_resData,
//                                                    depth, height, width, nn, a);
    }

    CUDA_CHECK_RETURN(cudaThreadSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());

    // Unbind texture
    CUDA_CHECK_RETURN(cudaUnbindTexture(t_data4));

    // Copy result into CPU memory
    CUDA_CHECK_RETURN(cudaMemcpy(h_resData, d_origData,
                                 sizeof(*d_origData) * dataSize,
                                 cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(d_origData));
    CUDA_CHECK_RETURN(cudaFree(d_resData));
}
