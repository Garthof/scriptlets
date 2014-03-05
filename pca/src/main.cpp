#include <cstdio>
#include <cstdlib>

#include <string>

#include <tmpformat.h>

#include "cudapca.h"

const std::string FILE_PATH = "/home/bzflamas/scriptlets/tmp-format/lena.tmp";
const int PATCH_RADIUS = 2;
const int NUM_PCA_DIMS = 3;

int
main(void) {
    const TmpFormat::TmpData tmpData = TmpFormat::loadFile(FILE_PATH);
    const int frames = tmpData.frames;
    const int width = tmpData.width;
    const int height = tmpData.height;
    const float *const data = tmpData.data;

    CUDAPCA::CUDAPCAData d_data =
            CUDAPCA::uploadData(frames, height, width, tmpData.data);
    CUDAPCA::CUDAPCAPatches d_patchData =
            CUDAPCA::generatePatches(d_data, PATCH_RADIUS);


    exit(EXIT_SUCCESS);
}
