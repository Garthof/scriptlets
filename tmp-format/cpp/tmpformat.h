#ifndef TMP_FORMAT_H
#define TMP_FORMAT_H

#include <string>

namespace TmpFormat {

    /// Class to store TMP file contents in memory.
    class TmpData {
    public:
        TmpData(const int frames, const int width,
                const int height, const int channels,
                const float *const data);

        ~TmpData();

    public:
        const int frames;
        const int width;
        const int height;
        const int channels;
        const float *const data;
    };

    /// Load file contents and generates a TmpData. Notice that the number
    /// of frames is interpreted as the depth in case of 3D volumes.
    TmpData
    loadFile(const std::string fileName);

    /// Saves TmpData. Notice that the number of frames is interpreted as
    /// the depth in case of 3D volumes.
    void
    saveFile(const std::string fileName, const TmpData &data);

};

#endif
