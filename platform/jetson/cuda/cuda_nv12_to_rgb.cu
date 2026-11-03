// Route2 CUDA preprocess: NV12 ER pitch → NCHW RGB (640×640 或 letterbox from 1080p).
// dst 为 TRT 输入 device buffer；无 CPU 拷贝。

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace route2 {

__device__ __forceinline__ float clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

__device__ __forceinline__ float sample_y_bilinear(
    const uint8_t* yPlane, int srcW, int srcH, int pitch, float fx, float fy) {
    fx = clampf(fx, 0.0f, static_cast<float>(srcW - 1));
    fy = clampf(fy, 0.0f, static_cast<float>(srcH - 1));
    const int x0 = static_cast<int>(fx);
    const int y0 = static_cast<int>(fy);
    const int x1 = min(x0 + 1, srcW - 1);
    const int y1 = min(y0 + 1, srcH - 1);
    const float ax = fx - static_cast<float>(x0);
    const float ay = fy - static_cast<float>(y0);
    const float v00 = static_cast<float>(yPlane[y0 * pitch + x0]);
    const float v01 = static_cast<float>(yPlane[y0 * pitch + x1]);
    const float v10 = static_cast<float>(yPlane[y1 * pitch + x0]);
    const float v11 = static_cast<float>(yPlane[y1 * pitch + x1]);
    return (1.0f - ax) * (1.0f - ay) * v00 + ax * (1.0f - ay) * v01
         + (1.0f - ax) * ay * v10 + ax * ay * v11;
}

__device__ __forceinline__ void sample_uv_bilinear(
    const uint8_t* uvPlane,
    int srcW,
    int srcH,
    int pitch,
    float fx,
    float fy,
    float& u,
    float& v) {
    const float uvFx = clampf(fx * 0.5f, 0.0f, static_cast<float>(srcW / 2 - 1));
    const float uvFy = clampf(fy * 0.5f, 0.0f, static_cast<float>(srcH / 2 - 1));
    const int x0 = static_cast<int>(uvFx);
    const int y0 = static_cast<int>(uvFy);
    const int x1 = min(x0 + 1, srcW / 2 - 1);
    const int y1 = min(y0 + 1, srcH / 2 - 1);
    const float ax = uvFx - static_cast<float>(x0);
    const float ay = uvFy - static_cast<float>(y0);
    const int stride = pitch;
    auto sample = [&](int cx, int cy) {
        const int col = cx << 1;
        const uint8_t* p = uvPlane + cy * stride + col;
        return make_float2(static_cast<float>(p[0]), static_cast<float>(p[1]));
    };
    const float2 p00 = sample(x0, y0);
    const float2 p01 = sample(x1, y0);
    const float2 p10 = sample(x0, y1);
    const float2 p11 = sample(x1, y1);
    u = (1.0f - ax) * (1.0f - ay) * p00.x + ax * (1.0f - ay) * p01.x
      + (1.0f - ax) * ay * p10.x + ax * ay * p11.x;
    v = (1.0f - ax) * (1.0f - ay) * p00.y + ax * (1.0f - ay) * p01.y
      + (1.0f - ax) * ay * p10.y + ax * ay * p11.y;
}

__device__ __forceinline__ void yuv_to_rgb_nchw(
    float Y, float U, float V, float* dstNchw, int planeSize, int idx, float scale) {
    U -= 128.0f;
    V -= 128.0f;
    const float R = clampf((Y + 1.402f * V) * scale, 0.0f, 1.0f);
    const float G = clampf((Y - 0.344f * U - 0.714f * V) * scale, 0.0f, 1.0f);
    const float B = clampf((Y + 1.772f * U) * scale, 0.0f, 1.0f);
    dstNchw[0 * planeSize + idx] = R;
    dstNchw[1 * planeSize + idx] = G;
    dstNchw[2 * planeSize + idx] = B;
}

__global__ void nv12ErPitchToNchwRgbKernel(
    const uint8_t* __restrict__ yPlane,
    const uint8_t* __restrict__ uvPlane,
    float* __restrict__ dstNchw,
    int width,
    int height,
    int pitch) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const float Y = static_cast<float>(yPlane[y * pitch + x]);
    const int uvRow = y >> 1;
    const int uvCol = (x >> 1) << 1;
    const float U = static_cast<float>(uvPlane[uvRow * pitch + uvCol]);
    const float V = static_cast<float>(uvPlane[uvRow * pitch + uvCol + 1]);

    const int plane = height * width;
    const int idx = y * width + x;
    const float scale = 1.0f / 255.0f;
    yuv_to_rgb_nchw(Y, U, V, dstNchw, plane, idx, scale);
}

__global__ void nv12ErLetterboxToNchwRgbKernel(
    const uint8_t* __restrict__ yPlane,
    const uint8_t* __restrict__ uvPlane,
    float* __restrict__ dstNchw,
    int srcW,
    int srcH,
    int pitch,
    int dstW,
    int dstH,
    int newW,
    int newH,
    int padLeft,
    int padTop,
    float padValue) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

    const int planeSize = dstH * dstW;
    const int idx = y * dstW + x;
    const int srcX = x - padLeft;
    const int srcY = y - padTop;

    if (srcX >= 0 && srcX < newW && srcY >= 0 && srcY < newH) {
        const float fx = static_cast<float>(srcX) * static_cast<float>(srcW)
                       / static_cast<float>(newW);
        const float fy = static_cast<float>(srcY) * static_cast<float>(srcH)
                       / static_cast<float>(newH);
        const float Y = sample_y_bilinear(yPlane, srcW, srcH, pitch, fx, fy);
        float U, V;
        sample_uv_bilinear(uvPlane, srcW, srcH, pitch, fx, fy, U, V);
        yuv_to_rgb_nchw(Y, U, V, dstNchw, planeSize, idx, 1.0f / 255.0f);
    } else {
        dstNchw[0 * planeSize + idx] = padValue;
        dstNchw[1 * planeSize + idx] = padValue;
        dstNchw[2 * planeSize + idx] = padValue;
    }
}

__global__ void nv12ErCropLetterboxToNchwRgbKernel(
    const uint8_t* __restrict__ yPlane,
    const uint8_t* __restrict__ uvPlane,
    float* __restrict__ dstNchw,
    int fullW,
    int fullH,
    int pitch,
    int cropLeft,
    int cropTop,
    int cropW,
    int cropH,
    int dstW,
    int dstH,
    int newW,
    int newH,
    int padLeft,
    int padTop,
    float padValue) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

    const int planeSize = dstH * dstW;
    const int idx = y * dstW + x;
    const int srcX = x - padLeft;
    const int srcY = y - padTop;

    if (srcX >= 0 && srcX < newW && srcY >= 0 && srcY < newH) {
        const float fx = static_cast<float>(cropLeft)
                       + static_cast<float>(srcX) * static_cast<float>(cropW)
                       / static_cast<float>(newW);
        const float fy = static_cast<float>(cropTop)
                       + static_cast<float>(srcY) * static_cast<float>(cropH)
                       / static_cast<float>(newH);
        const float Y = sample_y_bilinear(yPlane, fullW, fullH, pitch, fx, fy);
        float U, V;
        sample_uv_bilinear(uvPlane, fullW, fullH, pitch, fx, fy, U, V);
        yuv_to_rgb_nchw(Y, U, V, dstNchw, planeSize, idx, 1.0f / 255.0f);
    } else {
        dstNchw[0 * planeSize + idx] = padValue;
        dstNchw[1 * planeSize + idx] = padValue;
        dstNchw[2 * planeSize + idx] = padValue;
    }
}

__global__ void nv12ErStereoSbsCropLetterboxToNchwRgbKernel(
    const uint8_t* __restrict__ yPlane,
    const uint8_t* __restrict__ uvPlane,
    float* __restrict__ dstNchw0,
    float* __restrict__ dstNchw1,
    int fullW,
    int fullH,
    int pitch,
    int eyeW,
    int cropTop,
    int cropH,
    int dstW,
    int dstH,
    int newW,
    int newH,
    int padLeft,
    int padTop,
    float padValue) {
    const int batchIdx = blockIdx.z;
    if (batchIdx > 1) return;
    const int cropLeft = batchIdx * eyeW;
    float* __restrict__ dstNchw = batchIdx == 0 ? dstNchw0 : dstNchw1;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

    const int planeSize = dstH * dstW;
    const int idx = y * dstW + x;
    const int srcX = x - padLeft;
    const int srcY = y - padTop;

    if (srcX >= 0 && srcX < newW && srcY >= 0 && srcY < newH) {
        const float fx = static_cast<float>(cropLeft)
                       + static_cast<float>(srcX) * static_cast<float>(eyeW)
                       / static_cast<float>(newW);
        const float fy = static_cast<float>(cropTop)
                       + static_cast<float>(srcY) * static_cast<float>(cropH)
                       / static_cast<float>(newH);
        const float Y = sample_y_bilinear(yPlane, fullW, fullH, pitch, fx, fy);
        float U, V;
        sample_uv_bilinear(uvPlane, fullW, fullH, pitch, fx, fy, U, V);
        yuv_to_rgb_nchw(Y, U, V, dstNchw, planeSize, idx, 1.0f / 255.0f);
    } else {
        dstNchw[0 * planeSize + idx] = padValue;
        dstNchw[1 * planeSize + idx] = padValue;
        dstNchw[2 * planeSize + idx] = padValue;
    }
}

__device__ __forceinline__ void yuv_to_rgb_nchw_imagenet(
    float Y, float U, float V, float* dstNchw, int planeSize, int idx) {
    U -= 128.0f;
    V -= 128.0f;
    const float R = Y + 1.402f * V;
    const float G = Y - 0.344f * U - 0.714f * V;
    const float B = Y + 1.772f * U;
    dstNchw[0 * planeSize + idx] = (R - 123.675f) / 58.395f;
    dstNchw[1 * planeSize + idx] = (G - 116.28f) / 57.12f;
    dstNchw[2 * planeSize + idx] = (B - 103.53f) / 57.375f;
}

__device__ __forceinline__ float sample_map_bilinear(
    const float* mapPlane, int mapW, int mapH, float fx, float fy) {
    fx = clampf(fx, 0.0f, static_cast<float>(mapW - 1));
    fy = clampf(fy, 0.0f, static_cast<float>(mapH - 1));
    const int x0 = static_cast<int>(fx);
    const int y0 = static_cast<int>(fy);
    const int x1 = min(x0 + 1, mapW - 1);
    const int y1 = min(y0 + 1, mapH - 1);
    const float ax = fx - static_cast<float>(x0);
    const float ay = fy - static_cast<float>(y0);
    const float v00 = mapPlane[y0 * mapW + x0];
    const float v01 = mapPlane[y0 * mapW + x1];
    const float v10 = mapPlane[y1 * mapW + x0];
    const float v11 = mapPlane[y1 * mapW + x1];
    return (1.0f - ax) * (1.0f - ay) * v00 + ax * (1.0f - ay) * v01
         + (1.0f - ax) * ay * v10 + ax * ay * v11;
}

__global__ void nv12ErRemapCropLetterboxToNchwRgbKernel(
    const uint8_t* __restrict__ yPlane,
    const uint8_t* __restrict__ uvPlane,
    const float* __restrict__ mapX,
    const float* __restrict__ mapY,
    float* __restrict__ dstNchw,
    int fullW,
    int fullH,
    int pitch,
    int eyeCropLeft,
    int rectW,
    int rectH,
    int dstW,
    int dstH,
    int newW,
    int newH,
    int padLeft,
    int padTop,
    float padValue) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

    const int planeSize = dstH * dstW;
    const int idx = y * dstW + x;
    const int srcX = x - padLeft;
    const int srcY = y - padTop;

    if (srcX >= 0 && srcX < newW && srcY >= 0 && srcY < newH) {
        const float rx = static_cast<float>(srcX) * static_cast<float>(rectW)
                       / static_cast<float>(newW);
        const float ry = static_cast<float>(srcY) * static_cast<float>(rectH)
                       / static_cast<float>(newH);
        const float sx = sample_map_bilinear(mapX, rectW, rectH, rx, ry);
        const float sy = sample_map_bilinear(mapY, rectW, rectH, rx, ry);
        const float fx = static_cast<float>(eyeCropLeft) + sx;
        const float fy = sy;
        const float Y = sample_y_bilinear(yPlane, fullW, fullH, pitch, fx, fy);
        float U, V;
        sample_uv_bilinear(uvPlane, fullW, fullH, pitch, fx, fy, U, V);
        yuv_to_rgb_nchw(Y, U, V, dstNchw, planeSize, idx, 1.0f / 255.0f);
    } else {
        dstNchw[0 * planeSize + idx] = padValue;
        dstNchw[1 * planeSize + idx] = padValue;
        dstNchw[2 * planeSize + idx] = padValue;
    }
}

__global__ void nv12ErRemapBBoxLetterboxToNchwRgbImagenetKernel(
    const uint8_t* __restrict__ yPlane,
    const uint8_t* __restrict__ uvPlane,
    const float* __restrict__ mapX,
    const float* __restrict__ mapY,
    float* __restrict__ dstNchw,
    int fullW,
    int fullH,
    int pitch,
    int eyeCropLeft,
    int rectW,
    int rectH,
    int bboxLeft,
    int bboxTop,
    int bboxW,
    int bboxH,
    int dstW,
    int dstH,
    int newW,
    int newH,
    int padLeft,
    int padTop) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

    const int planeSize = dstH * dstW;
    const int idx = y * dstW + x;
    const int srcX = x - padLeft;
    const int srcY = y - padTop;

    if (srcX >= 0 && srcX < newW && srcY >= 0 && srcY < newH) {
        const float rx = static_cast<float>(bboxLeft)
                       + static_cast<float>(srcX) * static_cast<float>(bboxW)
                       / static_cast<float>(newW);
        const float ry = static_cast<float>(bboxTop)
                       + static_cast<float>(srcY) * static_cast<float>(bboxH)
                       / static_cast<float>(newH);
        const float sx = sample_map_bilinear(mapX, rectW, rectH, rx, ry);
        const float sy = sample_map_bilinear(mapY, rectW, rectH, rx, ry);
        const float fx = static_cast<float>(eyeCropLeft) + sx;
        const float fy = sy;
        const float Y = sample_y_bilinear(yPlane, fullW, fullH, pitch, fx, fy);
        float U, V;
        sample_uv_bilinear(uvPlane, fullW, fullH, pitch, fx, fy, U, V);
        yuv_to_rgb_nchw_imagenet(Y, U, V, dstNchw, planeSize, idx);
    } else {
        dstNchw[0 * planeSize + idx] = (0.0f - 123.675f) / 58.395f;
        dstNchw[1 * planeSize + idx] = (0.0f - 116.28f) / 57.12f;
        dstNchw[2 * planeSize + idx] = (0.0f - 103.53f) / 57.375f;
    }
}

__global__ void nv12ErCropLetterboxToNchwRgbImagenetKernel(
    const uint8_t* __restrict__ yPlane,
    const uint8_t* __restrict__ uvPlane,
    float* __restrict__ dstNchw,
    int fullW,
    int fullH,
    int pitch,
    int cropLeft,
    int cropTop,
    int cropW,
    int cropH,
    int dstW,
    int dstH,
    int newW,
    int newH,
    int padLeft,
    int padTop) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

    const int planeSize = dstH * dstW;
    const int idx = y * dstW + x;
    const int srcX = x - padLeft;
    const int srcY = y - padTop;

    if (srcX >= 0 && srcX < newW && srcY >= 0 && srcY < newH) {
        const float fx = static_cast<float>(cropLeft)
                       + static_cast<float>(srcX) * static_cast<float>(cropW)
                       / static_cast<float>(newW);
        const float fy = static_cast<float>(cropTop)
                       + static_cast<float>(srcY) * static_cast<float>(cropH)
                       / static_cast<float>(newH);
        const float Y = sample_y_bilinear(yPlane, fullW, fullH, pitch, fx, fy);
        float U, V;
        sample_uv_bilinear(uvPlane, fullW, fullH, pitch, fx, fy, U, V);
        yuv_to_rgb_nchw_imagenet(Y, U, V, dstNchw, planeSize, idx);
    } else {
        dstNchw[0 * planeSize + idx] = (0.0f - 123.675f) / 58.395f;
        dstNchw[1 * planeSize + idx] = (0.0f - 116.28f) / 57.12f;
        dstNchw[2 * planeSize + idx] = (0.0f - 103.53f) / 57.375f;
    }
}

}  // namespace route2

extern "C" int route2_nv12_er_pitch_to_nchw_rgb(
    const void* yPlane,
    const void* uvPlane,
    void* dstNchw,
    int width,
    int height,
    int pitch,
    cudaStream_t stream) {
    if (!yPlane || !uvPlane || !dstNchw || width <= 0 || height <= 0 || pitch < width) {
        return 1;
    }
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    route2::nv12ErPitchToNchwRgbKernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(yPlane),
        static_cast<const uint8_t*>(uvPlane),
        static_cast<float*>(dstNchw),
        width,
        height,
        pitch);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

extern "C" int route2_nv12_er_letterbox_to_nchw_rgb(
    const void* yPlane,
    const void* uvPlane,
    void* dstNchw,
    int srcW,
    int srcH,
    int pitch,
    int dstW,
    int dstH,
    int newW,
    int newH,
    int padLeft,
    int padTop,
    cudaStream_t stream) {
    if (!yPlane || !uvPlane || !dstNchw || srcW <= 0 || srcH <= 0 || pitch < srcW
        || dstW <= 0 || dstH <= 0 || newW <= 0 || newH <= 0) {
        return 1;
    }
    dim3 block(32, 8);
    dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);
    constexpr float padValue = 114.0f / 255.0f;
    route2::nv12ErLetterboxToNchwRgbKernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(yPlane),
        static_cast<const uint8_t*>(uvPlane),
        static_cast<float*>(dstNchw),
        srcW,
        srcH,
        pitch,
        dstW,
        dstH,
        newW,
        newH,
        padLeft,
        padTop,
        padValue);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

extern "C" int route2_nv12_er_crop_letterbox_to_nchw_rgb(
    const void* yPlane,
    const void* uvPlane,
    void* dstNchw,
    int fullW,
    int fullH,
    int pitch,
    int cropLeft,
    int cropTop,
    int cropW,
    int cropH,
    int dstW,
    int dstH,
    int newW,
    int newH,
    int padLeft,
    int padTop,
    cudaStream_t stream) {
    if (!yPlane || !uvPlane || !dstNchw || fullW <= 0 || fullH <= 0 || pitch < fullW
        || cropW <= 0 || cropH <= 0 || dstW <= 0 || dstH <= 0 || newW <= 0 || newH <= 0) {
        return 1;
    }
    dim3 block(32, 8);
    dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);
    constexpr float padValue = 114.0f / 255.0f;
    route2::nv12ErCropLetterboxToNchwRgbKernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(yPlane),
        static_cast<const uint8_t*>(uvPlane),
        static_cast<float*>(dstNchw),
        fullW,
        fullH,
        pitch,
        cropLeft,
        cropTop,
        cropW,
        cropH,
        dstW,
        dstH,
        newW,
        newH,
        padLeft,
        padTop,
        padValue);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

extern "C" int route2_nv12_er_stereo_sbs_crop_letterbox_to_nchw_rgb(
    const void* yPlane,
    const void* uvPlane,
    void* dstNchw0,
    void* dstNchw1,
    int fullW,
    int fullH,
    int pitch,
    int eyeW,
    int cropTop,
    int cropH,
    int dstW,
    int dstH,
    int newW,
    int newH,
    int padLeft,
    int padTop,
    cudaStream_t stream) {
    if (!yPlane || !uvPlane || !dstNchw0 || !dstNchw1 || fullW <= 0 || fullH <= 0
        || pitch < fullW || eyeW <= 0 || cropH <= 0 || dstW <= 0 || dstH <= 0
        || newW <= 0 || newH <= 0) {
        return 1;
    }
    dim3 block(32, 8);
    dim3 grid(
        (dstW + block.x - 1) / block.x,
        (dstH + block.y - 1) / block.y,
        2);
    constexpr float padValue = 114.0f / 255.0f;
    route2::nv12ErStereoSbsCropLetterboxToNchwRgbKernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(yPlane),
        static_cast<const uint8_t*>(uvPlane),
        static_cast<float*>(dstNchw0),
        static_cast<float*>(dstNchw1),
        fullW,
        fullH,
        pitch,
        eyeW,
        cropTop,
        cropH,
        dstW,
        dstH,
        newW,
        newH,
        padLeft,
        padTop,
        padValue);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

__device__ __forceinline__ void yuv_to_bgr_uint8(
    float Y, float U, float V, uint8_t* dstBgr, int idx) {
    U -= 128.0f;
    V -= 128.0f;
    const int R = static_cast<int>(fminf(fmaxf(Y + 1.402f * V, 0.0f), 255.0f));
    const int G = static_cast<int>(fminf(fmaxf(Y - 0.344f * U - 0.714f * V, 0.0f), 255.0f));
    const int B = static_cast<int>(fminf(fmaxf(Y + 1.772f * U, 0.0f), 255.0f));
    dstBgr[idx * 3 + 0] = static_cast<uint8_t>(B);
    dstBgr[idx * 3 + 1] = static_cast<uint8_t>(G);
    dstBgr[idx * 3 + 2] = static_cast<uint8_t>(R);
}

__global__ void nv12ErCropToBgrHwcKernel(
    const uint8_t* __restrict__ yPlane,
    const uint8_t* __restrict__ uvPlane,
    uint8_t* __restrict__ dstBgr,
    int fullW,
    int fullH,
    int pitch,
    int cropLeft,
    int cropTop,
    int cropW,
    int cropH) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cropW || y >= cropH) return;

    const int sx = cropLeft + x;
    const int sy = cropTop + y;
    const float Y = static_cast<float>(yPlane[sy * pitch + sx]);
    const int uvRow = sy >> 1;
    const int uvCol = (sx >> 1) << 1;
    const float U = static_cast<float>(uvPlane[uvRow * pitch + uvCol]);
    const float V = static_cast<float>(uvPlane[uvRow * pitch + uvCol + 1]);
    yuv_to_bgr_uint8(Y, U, V, dstBgr, y * cropW + x);
}

extern "C" int route2_nv12_er_crop_to_bgr_hwc(
    const void* yPlane,
    const void* uvPlane,
    void* dstBgr,
    int fullW,
    int fullH,
    int pitch,
    int cropLeft,
    int cropTop,
    int cropW,
    int cropH,
    cudaStream_t stream) {
    if (!yPlane || !uvPlane || !dstBgr || fullW <= 0 || fullH <= 0 || pitch < fullW
        || cropW <= 0 || cropH <= 0) {
        return 1;
    }
    dim3 block(32, 8);
    dim3 grid((cropW + block.x - 1) / block.x, (cropH + block.y - 1) / block.y);
    nv12ErCropToBgrHwcKernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(yPlane),
        static_cast<const uint8_t*>(uvPlane),
        static_cast<uint8_t*>(dstBgr),
        fullW,
        fullH,
        pitch,
        cropLeft,
        cropTop,
        cropW,
        cropH);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

extern "C" int route2_nv12_er_crop_letterbox_to_nchw_rgb_imagenet(
    const void* yPlane,
    const void* uvPlane,
    void* dstNchw,
    int fullW,
    int fullH,
    int pitch,
    int cropLeft,
    int cropTop,
    int cropW,
    int cropH,
    int dstW,
    int dstH,
    int newW,
    int newH,
    int padLeft,
    int padTop,
    cudaStream_t stream) {
    if (!yPlane || !uvPlane || !dstNchw || fullW <= 0 || fullH <= 0 || pitch < fullW
        || cropW <= 0 || cropH <= 0 || dstW <= 0 || dstH <= 0 || newW <= 0 || newH <= 0) {
        return 1;
    }
    dim3 block(32, 8);
    dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);
    route2::nv12ErCropLetterboxToNchwRgbImagenetKernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(yPlane),
        static_cast<const uint8_t*>(uvPlane),
        static_cast<float*>(dstNchw),
        fullW,
        fullH,
        pitch,
        cropLeft,
        cropTop,
        cropW,
        cropH,
        dstW,
        dstH,
        newW,
        newH,
        padLeft,
        padTop);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

extern "C" int route2_nv12_er_remap_crop_letterbox_to_nchw_rgb(
    const void* yPlane,
    const void* uvPlane,
    const void* mapX,
    const void* mapY,
    void* dstNchw,
    int fullW,
    int fullH,
    int pitch,
    int eyeCropLeft,
    int rectW,
    int rectH,
    int dstW,
    int dstH,
    int newW,
    int newH,
    int padLeft,
    int padTop,
    cudaStream_t stream) {
    if (!yPlane || !uvPlane || !mapX || !mapY || !dstNchw || fullW <= 0 || fullH <= 0
        || pitch < fullW || rectW <= 0 || rectH <= 0 || dstW <= 0 || dstH <= 0
        || newW <= 0 || newH <= 0) {
        return 1;
    }
    dim3 block(32, 8);
    dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);
    constexpr float padValue = 114.0f / 255.0f;
    route2::nv12ErRemapCropLetterboxToNchwRgbKernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(yPlane),
        static_cast<const uint8_t*>(uvPlane),
        static_cast<const float*>(mapX),
        static_cast<const float*>(mapY),
        static_cast<float*>(dstNchw),
        fullW,
        fullH,
        pitch,
        eyeCropLeft,
        rectW,
        rectH,
        dstW,
        dstH,
        newW,
        newH,
        padLeft,
        padTop,
        padValue);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

extern "C" int route2_nv12_er_remap_bbox_letterbox_to_nchw_rgb_imagenet(
    const void* yPlane,
    const void* uvPlane,
    const void* mapX,
    const void* mapY,
    void* dstNchw,
    int fullW,
    int fullH,
    int pitch,
    int eyeCropLeft,
    int rectW,
    int rectH,
    int bboxLeft,
    int bboxTop,
    int bboxW,
    int bboxH,
    int dstW,
    int dstH,
    int newW,
    int newH,
    int padLeft,
    int padTop,
    cudaStream_t stream) {
    if (!yPlane || !uvPlane || !mapX || !mapY || !dstNchw || fullW <= 0 || fullH <= 0
        || pitch < fullW || rectW <= 0 || rectH <= 0 || bboxW <= 0 || bboxH <= 0
        || dstW <= 0 || dstH <= 0 || newW <= 0 || newH <= 0) {
        return 1;
    }
    dim3 block(32, 8);
    dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);
    route2::nv12ErRemapBBoxLetterboxToNchwRgbImagenetKernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(yPlane),
        static_cast<const uint8_t*>(uvPlane),
        static_cast<const float*>(mapX),
        static_cast<const float*>(mapY),
        static_cast<float*>(dstNchw),
        fullW,
        fullH,
        pitch,
        eyeCropLeft,
        rectW,
        rectH,
        bboxLeft,
        bboxTop,
        bboxW,
        bboxH,
        dstW,
        dstH,
        newW,
        newH,
        padLeft,
        padTop);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

/** BGR HWC 90°：srcH×srcW → dst 为 srcW×srcH；mode 0=顺时针，1=逆时针。 */
__global__ void bgrHwcRotate90Kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int srcH,
    int srcW,
    int mode) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int dstH = srcW;
    const int dstW = srcH;
    if (x >= dstW || y >= dstH) return;

    int sx;
    int sy;
    if (mode == 0) {
        sx = y;
        sy = srcH - 1 - x;
    } else {
        sx = srcW - 1 - y;
        sy = x;
    }
    const int si = (sy * srcW + sx) * 3;
    const int di = (y * dstW + x) * 3;
    dst[di] = src[si];
    dst[di + 1] = src[si + 1];
    dst[di + 2] = src[si + 2];
}

extern "C" int route2_bgr_hwc_rotate90(
    const void* srcBgr,
    void* dstBgr,
    int srcH,
    int srcW,
    int mode,
    cudaStream_t stream) {
    if (!srcBgr || !dstBgr || srcH <= 0 || srcW <= 0 || (mode != 0 && mode != 1)) {
        return 1;
    }
    const int dstW = srcH;
    const int dstH = srcW;
    (void)dstH;
    dim3 block(32, 8);
    dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);
    bgrHwcRotate90Kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(srcBgr),
        static_cast<uint8_t*>(dstBgr),
        srcH,
        srcW,
        mode);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
