#ifndef CUDAPCA_H_
#define CUDAPCA_H_

#include <cstdlib>

namespace CUDAPCA {

    /// Basic data type.
    typedef float data_t;
    //typedef double data_t;

    /// Class to store PCA data contents in GPU memory.
    class CUDAPCAData {
    public:
        CUDAPCAData(const int depth,
                    const int height,
                    const int width,
                    const data_t *const data);

        ~CUDAPCAData();

    public:
        const data_t *const data;
        const int depth;
        const int height;
        const int width;
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
               const void *const h_data);

    /// Generate patches for each data element.
    CUDAPCAPatches
    generatePatches(const CUDAPCAData &d_data,
                    const int patchRadius);

    /// Compute eigenvectors from the patch space. Eigenvectors must be ordered
    /// by their eigenvalues (higher eigenvalues come first).
    CUDAPCAData
    generateEigenvecs(const CUDAPCAData &d_patches);

    /// Project patches into the first eigenvectors.
    CUDAPCAData
    projectPatches(const CUDAPCAData &d_data,
                   const CUDAPCAPatches &d_patches,
                   const int numPCADims);
}

#endif /* CUDAPCA_H_ */
