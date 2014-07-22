#ifndef SURF3DFILTER_H_
#define SURF3DFILTER_H_

void
computeResultFrom3DSurface(
        float4 *const h_resData,
        const float4 *const h_origData,
        const int depth, const int height, const int width,
        const float a);

#endif /* SURF3DFILTER_H_ */
