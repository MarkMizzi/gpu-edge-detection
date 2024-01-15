#pragma once

#include "gaussian_blur.cuh"

#include <imagebuf.hh>
#include <cudaimagebuf.cuh>
#include <utils.cuh>

template <typename PixelType>
__global__ void rgb_to_grayscale(PixelType *in,
                                 PixelType *out,
                                 ssize_t height,
                                 size_t pitch,
                                 size_t out_pitch)
{
    // we add one because we don't consider the edges of the image
    ssize_t x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    ssize_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;

    float r = CUDA_IMAGE_BUF_ACCESS(in, pitch, height, x, y, 0);
    float g = CUDA_IMAGE_BUF_ACCESS(in, pitch, height, x, y, 1);
    float b = CUDA_IMAGE_BUF_ACCESS(in, pitch, height, x, y, 2);

    float grayscale = 0.299 * r + 0.587 * g + 0.114 * b;
    CUDA_IMAGE_BUF_ACCESS(out, out_pitch, height, x, y, 0) = grayscale;
}

template <typename PixelType>
__global__ void sobel_grad_kern(PixelType *in,
                                PixelType *grad_buf,
                                ssize_t width,
                                ssize_t height,
                                size_t pitch,
                                size_t grad_buf_pitch,
                                float upper_threshold,
                                float lower_threshold)
{
    // we add one because we don't consider the edges of the image
    ssize_t x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    ssize_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;

    float grad = 0;

    /// Compute gradient in the x direction using a Sobel kernel
    float grad_x = 0;

    grad_x -= CUDA_IMAGE_BUF_ACCESS(in, pitch, height, x - 1, y - 1, 0);
    grad_x -= 2 * CUDA_IMAGE_BUF_ACCESS(in, pitch, height, x - 1, y, 0);
    grad_x -= CUDA_IMAGE_BUF_ACCESS(in, pitch, height, x - 1, y + 1, 0);

    grad_x += CUDA_IMAGE_BUF_ACCESS(in, pitch, height, x + 1, y - 1, 0);
    grad_x += 2 * CUDA_IMAGE_BUF_ACCESS(in, pitch, height, x + 1, y, 0);
    grad_x += CUDA_IMAGE_BUF_ACCESS(in, pitch, height, x + 1, y + 1, 0);

    /// Compute gradient in the y direction using a Sobel kernel
    float grad_y = 0;

    grad_y -= CUDA_IMAGE_BUF_ACCESS(in, pitch, height, x - 1, y - 1, 0);
    grad_y -= 2 * CUDA_IMAGE_BUF_ACCESS(in, pitch, height, x, y - 1, 0);
    grad_y -= CUDA_IMAGE_BUF_ACCESS(in, pitch, height, x + 1, y - 1, 0);

    grad_y += CUDA_IMAGE_BUF_ACCESS(in, pitch, height, x - 1, y + 1, 0);
    grad_y += 2 * CUDA_IMAGE_BUF_ACCESS(in, pitch, height, x, y + 1, 0);
    grad_y += CUDA_IMAGE_BUF_ACCESS(in, pitch, height, x - 1, y + 1, 0);

    // compute overall gradient, normalized to range [-1, 1]
    // The reason we divide by 4 * numeric_limits<PixelType>::max is bc with the Sobel
    // operators, largest possible value of grad_x and grad_y is this.
    grad += sqrt(grad_x * grad_x + grad_y * grad_y) / (numeric_limits<PixelType>::max * 4 * sqrt(2.0));

    // normalize gradient to range [0, 1]
    grad = (grad + 1) / 2;

    // apply thresholds
    grad *= (grad <= upper_threshold);
    grad *= (grad >= lower_threshold);

    CUDA_IMAGE_BUF_ACCESS(grad_buf, grad_buf_pitch, height, x, y, 0) = numeric_limits<PixelType>::max * grad;
}

template <typename PixelType>
ImageBuffer<PixelType> edge_detect(const ImageBuffer<PixelType> &image,
                                   float stddev,
                                   ssize_t rad_x,
                                   ssize_t rad_y,
                                   float upper_threshold,
                                   float lower_threshold,
                                   size_t width = 0,
                                   size_t height = 0)
{
    // set width and height to the defaults.
    width = width == 0 ? image.width : width;
    height = height == 0 ? image.height : height;

    // texture with normalized coords and normalized float read mode
    Layered2DTextureBuffer image_texbuf(image, true, true);
    CudaImageBuffer<PixelType> blurred_device(width, height, image.channels);
    CudaImageBuffer<PixelType> grayscale_device(width, height, 1);
    CudaImageBuffer<PixelType> grad_image_device(width, height, 1);

    // NOTE: Arbitrarily chosen. Adjust for performance.
    dim3 blocksize(16, 16);
    // we want division to round upwards, hence the -- trick.
    dim3 gridsize(
        -(-(ssize_t)image.width / blocksize.x),
        -(-(ssize_t)image.height / blocksize.y));

    ssize_t gauss_matrix_size = (2 * rad_x + 1) * (2 * rad_y + 1);

    gaussian_blur_kern<<<
        gridsize,
        blocksize,
        gauss_matrix_size * sizeof(float)>>>(
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

    rgb_to_grayscale<<<gridsize, blocksize>>>(
        blurred_device.data,
        grayscale_device.data,
        height,
        blurred_device.pitch / sizeof(PixelType),
        grayscale_device.pitch / sizeof(PixelType));

    checkCudaErrors(cudaGetLastError());

    // We don't consider the pixels at the edges when computing the gradient.
    gridsize = dim3((image.width - 2 + blocksize.x - 1) / blocksize.x,
                    (image.height - 2 + blocksize.y - 1) / blocksize.y);

    sobel_grad_kern<<<gridsize, blocksize>>>(
        grayscale_device.data,
        grad_image_device.data,
        width,
        height,
        blurred_device.pitch / sizeof(PixelType),
        grad_image_device.pitch / sizeof(PixelType),
        upper_threshold,
        lower_threshold);

    checkCudaErrors(cudaGetLastError());

    ImageBuffer<PixelType> grad_image(width, height, 1);
    grad_image_device.to_host_buffer(grad_image);

    return grad_image;
}