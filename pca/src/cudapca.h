#ifndef CUDAPCA_H_
#define CUDAPCA_H_

#include <cstdlib>

#include <memory>

namespace CUDAPCA {

#define CUDAPCA_USE_FLOAT

#ifdef CUDAPCA_USE_FLOAT
    typedef float data_t;
#else
    typedef double data_t;
#endif

    /// Class to store PCA data contents in GPU memory.
    class CUDAPCAData {
    public:
        CUDAPCAData(const int depth,
                    const int height,
                    const int width,
                    const data_t *const data);

    protected:
        CUDAPCAData(const int depth,
                    const int height,
                    const int width,
                    const size_t dataSize,
                    const data_t *const data);

    public:
        ~CUDAPCAData();

    private:
        void
        init(const size_t dataSize, const data_t *const data);

    public:
        const int depth;
        const int height;
        const int width;
        const data_t *const data;
    };

    /// Class to store PCA patches in GPU memory.
    class CUDAPCAPatches: public CUDAPCAData {
    public:
        CUDAPCAPatches(const int depth,
                       const int height,
                       const int width,
                       const int patchRadius,
                       const data_t *const data);

    public:
        const int patchRadius;
    };

    /// Upload data into GPU.
    CUDAPCAData
    uploadData(const int depth,
               const int height,
               const int width,
               const data_t *const h_data);

    /// Download data from GPU.
    std::auto_ptr<data_t>
    downloadData(const CUDAPCAData &d_data);

    /// Generate patches for each data element.
    CUDAPCAPatches
    generatePatches(const CUDAPCAData &d_data,
                    const int patchRadius);

    /// Download patches from GPU.
    std::auto_ptr<data_t>
    downloadPatches(const CUDAPCAPatches &d_patches);

    /// Compute eigenvectors from the patch space. Eigenvectors must be ordered
    /// by their eigenvalues (higher eigenvalues come first).
    CUDAPCAData
    generateEigenvecs(const CUDAPCAPatches &d_patches);

    /// Project patches into the first eigenvectors.
    CUDAPCAData
    projectPatches(const CUDAPCAData &d_data,
                   const CUDAPCAPatches &d_patches,
                   const int numPCADims);
}

#endif /* CUDAPCA_H_ */
