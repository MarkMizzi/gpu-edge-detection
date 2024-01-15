#pragma once

#include "gaussian_blur.cuh"

#include <imagebuf.hh>
#include <cudaimagebuf.cuh>
#include <utils.cuh>

template <typename PixelType>
__global__ void sobel_grad_kern(PixelType *in,
                                float *grad_buf,
                                ssize_t width,
                                ssize_t height,
                                size_t pitch,
                                size_t grad_buf_pitch)
{
    // we add one because we don't consider the edges of the image
    ssize_t x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    ssize_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;

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
    float grad = sqrt(grad_x * grad_x + grad_y * grad_y) / (numeric_limits<PixelType>::max * 4 * sqrt(2.0));
    float grad_direction = atan2(grad_y, grad_x);

    // normalize gradient to range [0, 1]
    grad = (grad + 1) / 2;

    CUDA_IMAGE_BUF_ACCESS(grad_buf, grad_buf_pitch, height, x, y, 0) = grad;
    CUDA_IMAGE_BUF_ACCESS(grad_buf, grad_buf_pitch, height, x, y, 1) = grad_direction;
}

template <typename PixelType>
__global__ void maximum_suppression_kern(float *grad_buf,
                                         PixelType *edges_image,
                                         ssize_t width,
                                         ssize_t height,
                                         size_t grad_buf_pitch,
                                         size_t edges_image_pitch,
                                         float upper_threshold,
                                         float lower_threshold)
{
    ssize_t x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    ssize_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;

    float grad = CUDA_IMAGE_BUF_ACCESS(grad_buf, grad_buf_pitch, height, x, y, 0);
    float grad_direction = CUDA_IMAGE_BUF_ACCESS(grad_buf, grad_buf_pitch, height, x, y, 1);

    // apply maximum suppression
    /**
     * Essentially we want the closest pixels in the direction of and opposite to the gradient vector.
     * For this we consider the following compass of direcions
     *
     * 3pi/4   pi/2   pi/4
     *    \     |     /
     *     \    |    /
     *      \   |   /
     * pi ------------- 0
     *      /   |   \
     *     /    |    \
     *    /     |     \
     * 5pi/4   3pi/2   7pi/4
     *
     * Anything with pi/4 <= grad_direction < 3pi/4 for example is compared to pixels with y - 1 and y + 1.
     * We can use bools as ints and some clever consideration of the compass above to
     *    figure out the ix and iy such that (x + ix, y + iy) and (x - ix, y - iy)
     *    are the pixels we need to compare with in the maximum suppression.
     */
    ssize_t iy = (grad_direction >= 5 * M_PI / 4 && grad_direction < 7 * M_PI / 4) -
                 (grad_direction >= M_PI / 4 && grad_direction < 3 * M_PI / 4);
    ssize_t ix = (grad_direction < M_PI / 4 || grad_direction >= 7 * M_PI / 4) -
                 (grad_direction >= 3 * M_PI / 4 && grad_direction < 5 * M_PI / 4);

    float last = CUDA_IMAGE_BUF_ACCESS(grad_buf, grad_buf_pitch, height, x - ix, y - iy, 0);
    float next = CUDA_IMAGE_BUF_ACCESS(grad_buf, grad_buf_pitch, height, x + ix, y + iy, 0);

    // This gradient value must be larger than the adjacent values on the
    //    line in the direction of the gradient.
    grad *= (grad >= last && grad >= next);

    // apply thresholds
    grad *= (grad <= upper_threshold);
    grad *= (grad >= lower_threshold);

    CUDA_IMAGE_BUF_ACCESS(edges_image, edges_image_pitch, height, x, y, 0) =
        numeric_limits<PixelType>::max * grad;
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

    ImageBuffer<PixelType> grayscale_image = image.to_grayscale();

    // texture with normalized coords and normalized float read mode
    Layered2DTextureBuffer grayscale_image_texbuf(grayscale_image, true, true);
    CudaImageBuffer<PixelType> blurred_device(width, height, image.channels);
    CudaImageBuffer<float> gradient_device(width, height, 2);
    CudaImageBuffer<PixelType> edges_image_device(width, height, 1);

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
        grayscale_image_texbuf.tex_obj,
        blurred_device.data,
        width,
        height,
        blurred_device.pitch / sizeof(PixelType),
        grayscale_image.channels,
        stddev,
        rad_x,
        rad_y);

    checkCudaErrors(cudaGetLastError());

    // We don't consider the pixels at the edges when computing the gradient.
    gridsize = dim3((image.width - 2 + blocksize.x - 1) / blocksize.x,
                    (image.height - 2 + blocksize.y - 1) / blocksize.y);

    sobel_grad_kern<<<gridsize, blocksize>>>(
        blurred_device.data,
        gradient_device.data,
        width,
        height,
        blurred_device.pitch / sizeof(PixelType),
        gradient_device.pitch / sizeof(float));

    checkCudaErrors(cudaGetLastError());

    maximum_suppression_kern<<<gridsize, blocksize>>>(
        gradient_device.data,
        edges_image_device.data,
        width,
        height,
        gradient_device.pitch / sizeof(float),
        edges_image_device.pitch / sizeof(PixelType),
        upper_threshold,
        lower_threshold);

    checkCudaErrors(cudaGetLastError());

    ImageBuffer<PixelType> edges_image(width, height, 1);
    edges_image_device.to_host_buffer(edges_image);

    return edges_image;
}