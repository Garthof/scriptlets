#ifndef TEX3DFILTER_H_
#define TEX3DFILTER_H_

void
computeResultFrom3DTexture(
        float4 *const h_resData,
        const float4 *const h_origData,
        const int depth, const int height, const int width,
        const float a);

#endif
