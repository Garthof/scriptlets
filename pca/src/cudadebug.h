#ifndef CUDADEBUG_H
#define CUDADEBUG_H

#include <sstream>
#include <string>

#include <tmpformat.h>

template <typename T>
inline std::string
toString(const T &t)
{
    std::stringstream ss;
    ss << t;
    return ss.str();
}


inline void
saveGPUBuffer(
        const std::string fileName,
        const int depth, const int width, const int height,
        const thrust::device_vector<CUDAPCA::data_t> &d_data)
{
    const thrust::host_vector<float> h_data(d_data);

    const TmpFormat::TmpData t_data(depth, width, height, 1, h_data.data());
    TmpFormat::saveFile(fileName + toString(".tmp"), t_data);
}

#endif
