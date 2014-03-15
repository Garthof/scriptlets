#include <cstdio>
#include <cstdlib>

#include <string>
#include <vector>

#include <tmpformat.h>

#include "cudapca.h"
#include "eigenvectors.h"

const std::string FILE_PATH = "/home/bzflamas/scriptlets/pca/test3x3x3.tmp";
const int PATCH_RADIUS = 2;
const int NUM_PCA_DIMS = 3;


std::vector<CUDAPCA::data_t>
getDataFromTmp(const TmpFormat::TmpData &tmpData)
{
    const int frames = tmpData.frames;
    const int width = tmpData.width;
    const int height = tmpData.height;
    const float *const inData = tmpData.data;

    const int dataSize = frames * width * height;

    std::vector<CUDAPCA::data_t> outData;
    outData.reserve(dataSize);

    for (int i = 0; i < dataSize; i++) {
        outData.push_back(inData[i]);
    }

    return outData;
}


TmpFormat::TmpData
getTmpFromData(const int frames, const int width,
               const int height, const int channels,
               const std::vector<CUDAPCA::data_t> &inData)
{
    typedef std::vector<CUDAPCA::data_t>::const_iterator iter_t;

    std::vector<float> outData;

    for (iter_t i = inData.begin(); i != inData.end(); i++) {
        outData.push_back(*i);
    }

    return TmpFormat::TmpData(frames, width, height, channels,
                              outData.data());
}


int
main(void) {
    const TmpFormat::TmpData tmpData = TmpFormat::loadFile(FILE_PATH);
    const int frames = tmpData.frames;
    const int width = tmpData.width;
    const int height = tmpData.height;
    const float *const data = tmpData.data;

    // Test data
    const std::vector<CUDAPCA::data_t> t_data = getDataFromTmp(tmpData);
    CUDAPCA::CUDAPCAData d_data =
            CUDAPCA::uploadData(frames, height, width, t_data);

    const std::vector<CUDAPCA::data_t> h_data = CUDAPCA::downloadData(d_data);


    const TmpFormat::TmpData t_tmpData =
            getTmpFromData(frames, width, height, 1, h_data);

    TmpFormat::saveFile("t_data.tmp", t_tmpData);

    // Test patches
    CUDAPCA::CUDAPCAPatches d_patches =
            CUDAPCA::generatePatches(d_data, PATCH_RADIUS);
    const int patchDiam = (2 * d_patches.patchRadius + 1);
    const int patchSize = patchDiam * patchDiam * patchDiam;

    const std::vector<CUDAPCA::data_t> h_patches =
            CUDAPCA::downloadPatches(d_patches);
    const TmpFormat::TmpData t_tmpPatches =
            getTmpFromData(1, patchSize, frames * width * height, 1, h_patches);

    TmpFormat::saveFile("t_patches.tmp", t_tmpPatches);

    // Test eigenvectors
    const CUDAPCA::CUDAPCAData d_eigenvecs =
            CUDAPCA::generateEigenvecs(d_patches, NUM_PCA_DIMS);

    const std::vector<CUDAPCA::data_t> h_eigenvecs =
            CUDAPCA::downloadData(d_eigenvecs);
    const TmpFormat::TmpData t_eigenvecs =
            getTmpFromData(d_eigenvecs.depth, d_eigenvecs.width,
                           d_eigenvecs.height, 1, h_eigenvecs);

    TmpFormat::saveFile("t_eigenvecs.tmp", t_eigenvecs);

    exit(EXIT_SUCCESS);
}
