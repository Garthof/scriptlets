#define CUDA_CHECK_RETURN(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        fprintf(stderr, "Error %s at line %d in file %s\n",                 \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);       \
        exit(1);                                                            \
    } }


template <typename T>
#ifdef __CUDA_ARCH__
__host__ __device__ __forceinline__
#else
inline
#endif
T
ceil_div(const T &a, const T &b)
{
    return ((a + b - 1) / b);
}
