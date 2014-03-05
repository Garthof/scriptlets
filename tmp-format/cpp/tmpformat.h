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
        const float *const data;
        const int frames;
        const int width;
        const int height;
        const int channels;
    };

    /// Load file contents and generates a TmpData.
    TmpData
    loadFile(const std::string fileName);

    /// Saves TmpData.
    void
    saveFile(const std::string fileName, const TmpData &data);

};

#endif
