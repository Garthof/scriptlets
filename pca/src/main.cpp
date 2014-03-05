#include <cstdio>
#include <cstdlib>

#include <memory>
#include <string>

#include <tmpformat.h>

#include "cudapca.h"

const std::string FILE_PATH = "/home/bzflamas/scriptlets/pca/test3x3x3.tmp";
const int PATCH_RADIUS = 2;
const int NUM_PCA_DIMS = 3;

int
main(void) {
    const TmpFormat::TmpData tmpData = TmpFormat::loadFile(FILE_PATH);
    const int frames = tmpData.frames;
    const int width = tmpData.width;
    const int height = tmpData.height;
    const float *const data = tmpData.data;

    // Test data
    CUDAPCA::CUDAPCAData d_data =
            CUDAPCA::uploadData(frames, height, width, tmpData.data);

    const std::auto_ptr<CUDAPCA::data_t> h_data =
            CUDAPCA::downloadData(d_data);
    const TmpFormat::TmpData t_tmpData(frames, width, height, 1,
                                       h_data.get());

    TmpFormat::saveFile("t_data.tmp", t_tmpData);

    // Test patches
    CUDAPCA::CUDAPCAPatches d_patches =
            CUDAPCA::generatePatches(d_data, PATCH_RADIUS);
    const int patchDiam = (2 * d_patches.patchRadius + 1);
    const int patchSize = patchDiam * patchDiam * patchDiam;

    const std::auto_ptr<CUDAPCA::data_t> h_patches =
            CUDAPCA::downloadPatches(d_patches);
    const TmpFormat::TmpData t_tmpPatches(1, patchSize,
                                          frames * width * height, 1,
                                          h_patches.get());

    TmpFormat::saveFile("t_patches.tmp", t_tmpPatches);


    exit(EXIT_SUCCESS);
}
