namespace Solution3 {

constexpr int BLOCK_DIM = 16;
constexpr int BLOCK_SIZE = BLOCK_DIM * BLOCK_DIM;

#define SCAN_BLOCK_DIM BLOCK_SIZE
#include "exclusiveScan.cu_inl"

__global__ void kernelRenderCircles() {
  __shared__ uint circleIsInBox[BLOCK_SIZE];
  __shared__ uint circleIndex[BLOCK_SIZE];
  __shared__ uint scratch[2 * BLOCK_SIZE];
  __shared__ int inBoxCircles[BLOCK_SIZE];

  int boxL = blockIdx.x * BLOCK_DIM;
  int boxB = blockIdx.y * BLOCK_DIM;
  int boxR = min(boxL + BLOCK_DIM, cuConstRendererParams.imageWidth);
  int boxT = min(boxB + BLOCK_DIM, cuConstRendererParams.imageHeight);
  float boxLNorm = boxL * cuConstRendererParams.invWidth;
  float boxRNorm = boxR * cuConstRendererParams.invWidth;
  float boxTNorm = boxT * cuConstRendererParams.invHeight;
  float boxBNorm = boxB * cuConstRendererParams.invHeight;

  int index = threadIdx.y * BLOCK_DIM + threadIdx.x;
  int pixelX = boxL + threadIdx.x;
  int pixelY = boxB + threadIdx.y;
  int pixelId = pixelY * cuConstRendererParams.imageWidth + pixelX;

  for (int i = 0; i < cuConstRendererParams.numCircles; i += BLOCK_SIZE) {
    int circleId = i + index;
    if (circleId < cuConstRendererParams.numCircles) {
      float3 p = *reinterpret_cast<float3 *>(
          &cuConstRendererParams.position[3 * circleId]);
      circleIsInBox[index] =
          circleInBox(p.x, p.y, cuConstRendererParams.radius[circleId],
                      boxLNorm, boxRNorm, boxTNorm, boxBNorm);
    } else {
      circleIsInBox[index] = 0;
    }
    __syncthreads();

    sharedMemExclusiveScan(index, circleIsInBox, circleIndex, scratch,
                           BLOCK_SIZE);
    if (circleIsInBox[index]) {
      inBoxCircles[circleIndex[index]] = circleId;
    }
    __syncthreads();

    int numCirclesInBox =
        circleIndex[BLOCK_SIZE - 1] + circleIsInBox[BLOCK_SIZE - 1];
    __syncthreads();

    if (pixelX < boxR && pixelY < boxT) {
      float4 *imgPtr = reinterpret_cast<float4 *>(
          &cuConstRendererParams.imageData[4 * pixelId]);
      for (int j = 0; j < numCirclesInBox; j++) {
        circleId = inBoxCircles[j];
        shadePixel(
            circleId,
            make_float2((pixelX + 0.5) * cuConstRendererParams.invWidth,
                        (pixelY + 0.5) * cuConstRendererParams.invHeight),
            *reinterpret_cast<float3 *>(
                &cuConstRendererParams.position[3 * circleId]),
            imgPtr);
      }
    }
  }
}

void renderCircles(int width, int height) {
  kernelRenderCircles<<<dim3((width + BLOCK_DIM - 1) / BLOCK_DIM,
                             (height + BLOCK_DIM - 1) / BLOCK_DIM),
                        dim3(BLOCK_DIM, BLOCK_DIM)>>>();
  cudaCheckError(cudaDeviceSynchronize());
}
} // namespace Solution3
