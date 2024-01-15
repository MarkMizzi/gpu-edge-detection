#pragma once

#include <imagebuf.hh>
#include <cudaimagebuf.cuh>
#include <texbuf.cuh>
#include <utils.cuh>

#include <cmath>

template <typename PixelType>
__global__ void gaussian_blur_kern(cudaTextureObject_t image,
                                   PixelType *blurred,
                                   ssize_t width,
                                   ssize_t height,
                                   size_t pitch,
                                   size_t channels,
                                   float stddev,
                                   ssize_t rad_x,
                                   ssize_t rad_y)
{
    ssize_t x = blockIdx.x * blockDim.x + threadIdx.x;
    ssize_t y = blockIdx.y * blockDim.y + threadIdx.y;

    for (size_t z = 0; z < channels; z++)
    {
        float res = 0;

        for (ssize_t dx = -rad_x; dx < rad_x; dx++)
        {
            for (ssize_t dy = -rad_y; dy < rad_y; dy++)
            {
                float norm_x = (x + dx) / (float)width;
                float norm_y = (y + dy) / (float)height;

                float val = tex2DLayered<float>(image, norm_x, norm_y, z);

                res += val * exp(-(dx * dx + dy * dy) / (2 * stddev * stddev)) /
                       (2 * M_PI * stddev * stddev);
            }
        }

        CUDA_IMAGE_BUF_ACCESS(blurred, pitch, height, x, y, z) = numeric_limits<PixelType>::max() * res;
    }
}

template <typename PixelType>
ImageBuffer<PixelType> gaussian_blur(const ImageBuffer<PixelType> &image,
                                     float stddev,
                                     ssize_t rad_x,
                                     ssize_t rad_y,
                                     size_t width = 0,
                                     size_t height = 0)
{
    // set width and height to the defaults.
    width = width == 0 ? image.width : width;
    height = height == 0 ? image.height : height;

    // texture with normalized coords and normalized float read mode
    Layered2DTextureBuffer image_texbuf(image, true, true);
    CudaImageBuffer<PixelType> blurred_device(width, height, image.channels);

    // NOTE: Arbitrarily chosen. Adjust for performance.
    dim3 blocksize(16, 16);
    // we want division to round upwards, hence the -- trick.
    dim3 gridsize(
        -(-(ssize_t)image.width / blocksize.x),
        -(-(ssize_t)image.height / blocksize.y));

    gaussian_blur_kern<<<gridsize, blocksize>>>(
        image_texbuf.tex_obj,
        blurred_device.data,
        width,
        height,
        blurred_device.pitch / sizeof(PixelType),
        image.channels,
        stddev,
        rad_x,
        rad_y);

    checkCudaErrors(cudaGetLastError());

    ImageBuffer<PixelType> blurred(width, height, image.channels);
    blurred_device.to_host_buffer(blurred);

    return blurred;
}