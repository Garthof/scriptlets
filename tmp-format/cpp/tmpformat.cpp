#include "tmpformat.h"

#include <fstream>
#include <vector>

#include <cstdlib>
#include <cstring>


TmpFormat::TmpData
parseData(const char *const rawData);

// TmpFormat::TmpData members

TmpFormat::TmpData::TmpData(
        const int _frames, const int _width,
        const int _height, const int _channels,
        const float *_data)
    : data(_data)
    , frames(_frames)
    , width(_width)
    , height(_height)
    , channels(_channels)
{

}


TmpFormat::TmpData::~TmpData()
{
    delete[] data;
}


// TmpFormat functions

TmpFormat::TmpData
TmpFormat::loadFile(const std::string fileName)
{
    // Open file
    std::ifstream file;
    file.open(fileName.c_str(),
              std::ios::in | std::ios::binary | std::ios::ate);

    // Get file size and store contents in memory
    const std::streampos fileSize = file.tellg();
    std::vector<char> rawData(fileSize);

    file.seekg(0, std::ios::beg);
    file.read(rawData.data(),
              static_cast<long int>(fileSize));

    // Clean and return
    file.close();
    return parseData(rawData.data());
}


// Auxiliary functions

TmpFormat::TmpData
parseData(const char *const rawData)
{
    // Read the metadata in header
    const int *const headerData =
            reinterpret_cast<const int *>(rawData);

    int h = 0;
    const int frames = headerData[h++];
    const int width = headerData[h++];
    const int height = headerData[h++];
    const int channels = headerData[h++];

    // Read the rest of the data and store it in a new buffer
    const float *const inputData =
            reinterpret_cast<const float *>(&headerData[h++]);

    const size_t dataSize = frames * width * height * channels;
    float *const outputData = new float[dataSize];

    memcpy(outputData, inputData, dataSize * sizeof(*outputData));

    // Generate new TmpData instance and return it
    return TmpFormat::TmpData(frames, width, height, channels,
                              outputData);
}


// Main for testing purposes.

int
main()
{
    TmpFormat::TmpData data = TmpFormat::loadFile("../lena.tmp");

    exit(EXIT_SUCCESS);
}