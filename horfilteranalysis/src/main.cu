#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <vector>

#include "constants.h"
#include "HorFilterAnalysis.h"
#include "TextureFilterAnalysis.h"
#include "SharedMemoryFilterAnalysis.h"
#include "BlockFilter.h"
#include "Tex3DFilter.h"
#include "Surf3DFilter.h"
#include "PrintMemory.h"

bool
compareResults(const std::vector<float> &a, const std::vector<float> &b)
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
    const int depth = WORK_SIZE_DEPTH;
    const int height = WORK_SIZE_HEIGHT;
    const int width = WORK_SIZE_WIDTH;
    const int nn = 4;
    const int dataSize = depth * height * width * nn;

//    printText();
//    return 0;

    // Generate data in CPU
    std::vector<float> h_origData(dataSize);
    for (int i = 0; i < dataSize; i++) {
        const float r = float(std::rand()) / float(RAND_MAX);
        h_origData[i] = r - 0.5f;
    }

    // Copy data into GPU
    std::vector<float> h_resData(dataSize);
    computeResult(&h_resData[0], &h_origData[0], depth, height, width, nn, a);

    // Use vector types
    if (nn == 2)  {
        const std::vector<float> h_origData2 = h_origData;
        std::vector<float> h_resData2(dataSize);
        computeResult((float2 *) &h_resData2[0], (float2 *) &h_origData2[0],
                      depth, height, width, 1, a);

        if (compareResults(h_resData, h_resData2)) {
            std::cout << "Results are equal" << std::endl;
        } else {
            std::cerr << "Results are different" << std::endl;
        }
    }

    if (nn == 3)  {
        const std::vector<float> h_origData3 = h_origData;
        std::vector<float> h_resData3(dataSize);
        computeResult((float3 *) &h_resData3[0], (float3 *) &h_origData3[0],
                      depth, height, width, 1, a);

        if (compareResults(h_resData, h_resData3)) {
            std::cout << "Results are equal" << std::endl;
        } else {
            std::cerr << "Results are different" << std::endl;
        }
    }

    if (nn == 4)  {
        std::cout << "Use global memory: ";

        const std::vector<float> h_origData4 = h_origData;
        std::vector<float> h_resData4(dataSize);
        computeResult((float4 *) &h_resData4[0], (float4 *) &h_origData4[0],
                      depth, height, width, 1, a);

        if (compareResults(h_resData, h_resData4)) {
            std::cout << "Results are equal" << std::endl;
        } else {
            std::cerr << "Results are different" << std::endl;
        }
    }

    // Use texture types
//    {
//        std::vector<float> h_resDataTex(dataSize);
//        computeResultFromTexture<nn>(&h_resDataTex[0],
//                                     &h_origData[0],
//                                     depth, height, width, a);
//
//        if (compareResults(h_resData, h_resDataTex)) {
//            std::cout << "Results are equal" << std::endl;
//        } else {
//            std::cerr << "Results are different" << std::endl;
//        }
//    }

    if (nn == 4)  {
        std::cout << "Use 1D texture: ";

        const std::vector<float> h_origData4 = h_origData;
        std::vector<float> h_resData4(dataSize);
        computeResultFromTexture((float4 *) &h_resData4[0],
                                 (float4 *) &h_origData4[0],
                                 depth, height, width, 1, a);

        if (compareResults(h_resData, h_resData4)) {
            std::cout << "Results are equal" << std::endl;
        } else {
            std::cerr << "Results are different" << std::endl;
        }
    }

    if (nn == 4)  {
        std::cout << "Use 1D texture and SM: ";

        const std::vector<float> h_origData4 = h_origData;
        std::vector<float> h_resData4(dataSize);
        computeResultFromTextureSM((float4 *) &h_resData4[0],
                                   (float4 *) &h_origData4[0],
                                   depth, height, width, 1, a);

        if (compareResults(h_resData, h_resData4)) {
            std::cout << "Results are equal" << std::endl;
        } else {
            std::cerr << "Results are different" << std::endl;
        }
    }

    if (nn == 4)  {
        std::cout << "Use 3D texture: ";

        const std::vector<float> h_origData4 = h_origData;
        std::vector<float> h_resData4(dataSize);
        computeResultFrom3DTexture((float4 *) &h_resData4[0],
                                   (float4 *) &h_origData4[0],
                                   depth, height, width, a);

        if (compareResults(h_resData, h_resData4)) {
            std::cout << "Results are equal" << std::endl;
        } else {
            std::cerr << "Results are different" << std::endl;
        }
    }

    if (nn == 4)  {
        std::cout << "Use 3D surface: ";

        const std::vector<float> h_origData4 = h_origData;
        std::vector<float> h_resData4(dataSize);
        computeResultFrom3DSurface((float4 *) &h_resData4[0],
                                   (float4 *) &h_origData4[0],
                                   depth, height, width, a);

        if (compareResults(h_resData, h_resData4)) {
            std::cout << "Results are equal" << std::endl;
        } else {
            std::cerr << "Results are different" << std::endl;
        }
    }

    // Use blockwise approach
    if (nn == 4) {
        std::cout << "Use blocks: ";

        const std::vector<float> h_origData4 = h_origData;
        std::vector<float> h_resData4(dataSize);
        computeBlockResultFromTexture((float4 *) &h_resData4[0],
                                      (float4 *) &h_origData4[0],
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
