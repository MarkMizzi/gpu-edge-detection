#pragma once

#include "utils.cuh"
#include "imagebuf.hh"

#include <cassert>

#define CUDA_IMAGE_BUF_ACCESS(BUFDATA, PITCH, YS, X, Y, Z) \
    (BUFDATA)[(Z) * (PITCH) * (YS) + (Y) * (PITCH) + (X)]

template <typename PixelType>
class CudaImageBuffer
{
public:
    const size_t width, height, channels;

    // This pitch is in bytes, UNLIKE the one in ImageBuffer.
    size_t pitch;
    PixelType *data;

    void to_host_buffer(ImageBuffer<PixelType> &host_buf) const
    {
        assert(host_buf.width == width);
        assert(host_buf.height == height);
        assert(host_buf.channels == channels);

        cudaExtent extent = make_cudaExtent(width * sizeof(PixelType),
                                            height,
                                            channels);
        // copy data from 3D linear memory
        cudaMemcpy3DParms copyParams = {0};
        copyParams.dstPtr = make_cudaPitchedPtr((void *)host_buf.data,
                                                host_buf.pitch * sizeof(PixelType),
                                                host_buf.width * sizeof(PixelType),
                                                host_buf.height);
        copyParams.srcPtr = make_cudaPitchedPtr((void *)data,
                                                pitch,
                                                width * sizeof(PixelType),
                                                height);
        copyParams.extent = extent;
        copyParams.kind = cudaMemcpyDeviceToHost;
        checkCudaErrors(cudaMemcpy3D(&copyParams));
    }

    CudaImageBuffer(const ImageBuffer<PixelType> &buf)
        : width(buf.width),
          height(buf.height),
          channels(buf.channels)
    {
        cudaExtent extent = make_cudaExtent(width * sizeof(PixelType),
                                            height,
                                            channels);

        cudaPitchedPtr pitched_ptr = {0};
        checkCudaErrors(cudaMalloc3D(&pitched_ptr, extent));

        data = (PixelType *)pitched_ptr.ptr;
        pitch = pitched_ptr.pitch;

        // copy data to 3D linear memory
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr = make_cudaPitchedPtr((void *)buf.data,
                                                buf.pitch * sizeof(PixelType),
                                                buf.width * sizeof(PixelType),
                                                buf.height);
        copyParams.dstPtr = make_cudaPitchedPtr((void *)data,
                                                pitch,
                                                width * sizeof(PixelType),
                                                height);
        copyParams.extent = extent;
        copyParams.kind = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));
    }

    CudaImageBuffer(size_t width, size_t height, size_t channels)
        : width(width),
          height(height),
          channels(channels)
    {

        cudaExtent extent = make_cudaExtent(width * sizeof(PixelType),
                                            height,
                                            channels);

        cudaPitchedPtr pitched_ptr;
        checkCudaErrors(cudaMalloc3D(&pitched_ptr, extent));

        data = (PixelType *)pitched_ptr.ptr;
        pitch = pitched_ptr.pitch;

        // zero out memory
        checkCudaErrors(cudaMemset3D(pitched_ptr, 0, extent));
    }

    ~CudaImageBuffer()
    {
        checkCudaErrors(cudaFree(data));
    }
};