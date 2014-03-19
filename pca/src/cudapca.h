#ifndef CUDAPCA_H_
#define CUDAPCA_H_

#include <cstdlib>

#include <vector>

namespace CUDAPCA {

#define CUDAPCA_USE_FLOAT

#ifdef CUDAPCA_USE_FLOAT
    typedef float data_t;
#else
    typedef double data_t;
#endif

    /// Class to store PCA data contents in GPU memory.
    class CUDAPCAData {
    // Constructors and destructor
    public:
        CUDAPCAData(const int depth,
                    const int height,
                    const int width,
                    const data_t *const data);

    protected:
        CUDAPCAData(const int depth,
                    const int height,
                    const int width,
                    const int dataSize,
                    const data_t *const data);

    public:
        virtual ~CUDAPCAData();

    // Function members
    public:
        inline const data_t*
        data() const { return dataBuffer; }

    private:
        data_t *
        initData(const int dataSize, const data_t *const data);

    // Variable members
    public:
        const int depth;
        const int height;
        const int width;

    private:
        data_t *const dataBuffer;
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
               const std::vector<data_t> &h_data);

    /// Download data from GPU.
    std::vector<data_t>
    downloadData(const CUDAPCAData &d_data);

    /// Generate patches for each data element.
    CUDAPCAPatches
    generatePatches(const CUDAPCAData &d_data,
                    const int patchRadius);

    /// Download patches from GPU.
    std::vector<data_t>
    downloadPatches(const CUDAPCAPatches &d_patches);

    /// Compute eigenvectors from the patch space. Eigenvectors must be ordered
    /// by their eigenvalues (higher eigenvalues come first).
    CUDAPCAData
    generateEigenvecs(const CUDAPCAPatches &d_patches,
                      const int numPCADims);

    /// Project patches into the first eigenvectors.
    CUDAPCAData
    projectPatches(const CUDAPCAPatches &d_patches,
                   const CUDAPCAData &d_eigenvecs);
}

#endif /* CUDAPCA_H_ */
