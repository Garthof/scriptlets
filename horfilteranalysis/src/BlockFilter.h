#ifndef BLOCKFILTER_H_
#define BLOCKFILTER_H_

void
computeBlockResultFromTexture(
        float4 *const h_resData,
        const float4 *const h_origData,
        const int depth, const int height, const int width,
        const float a);

#endif /* BLOCKFILTER_H_ */
