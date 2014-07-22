#ifndef PRINTMEMORY_H_
#define PRINTMEMORY_H_

texture<float, cudaTextureType1D> t_tex;

__global__ void
kernPrintTex(const int size)
{
    printf("Texture: ");

    for (int i = 0; i < size; i++) {
        printf("%f, ", tex1Dfetch(t_tex, i));
    }

    printf("\n");
}


__global__ void
kernPrintData(const float *const g_data, const int size)
{
    printf("Data   : ");

    for (int i = 0; i < size; i++) {
        printf("%f, ", g_data[i]);
    }

    printf("\n");
}


void
printText()
{
    float h_data[10] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f };
    float *d_data;

    cudaSetDevice(0);

    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_data, 10 * sizeof(*d_data)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_data, h_data, 10 * sizeof(*d_data),
                                 cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaBindTexture(0, t_tex, d_data, 10 * sizeof(*d_data)));
//
//    cudaArray_t a_data;
//    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
//    cudaExtent extent = make_cudaExtent(10, 1, 1);
//    CUDA_CHECK_RETURN(cudaMalloc3DArray(&a_data, &desc, extent));
//
//    cudaMemcpy3DParms copyParms = {0};
//    copyParms.srcPtr = make_cudaPitchedPtr(h_data, 0, 10, 1);
//    copyParms.dstArray = a_data;
//    copyParms.extent = extent;
//    copyParms.kind = cudaMemcpyHostToDevice;
//    CUDA_CHECK_RETURN(cudaMemcpy3D(&copyParms));
//
//    CUDA_CHECK_RETURN(cudaBindTextureToArray(t_tex, a_data));

    const dim3 dimBlk(1);
    const dim3 dimGrd(1);

    kernPrintTex<<<dimGrd, dimBlk>>>(10);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());

    CUDA_CHECK_RETURN(cudaUnbindTexture(t_tex));

    kernPrintData<<<dimGrd, dimBlk>>>(d_data, 10);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());
    CUDA_CHECK_RETURN(cudaUnbindTexture(t_tex));
}


#endif /* PRINTMEMORY_H_ */
