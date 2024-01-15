#pragma once

#include "utils.cuh"
#include "imagebuf.hh"

#include <cstring>
#include <type_traits>

/** ChannelFormatDescConstructor is used to generate code that constructs a cudaChannelFormatDesc
 * at compile-time depending on the PixelType used.
 *
 * There are multiple versions of the template, depending on whether PixelType is a float, a signed int,
 * or an unsigned int.
 */
template <typename PixelType,
          typename std::enable_if<std::is_floating_point<PixelType>{}, bool>::type = true>
inline cudaChannelFormatDesc get_channel_format_desc()
{
    return cudaCreateChannelDesc(sizeof(PixelType) * 8, 0, 0, 0, cudaChannelFormatKindFloat);
}

template <typename PixelType,
          typename std::enable_if<(!std::is_floating_point<PixelType>{} && std::is_signed<PixelType>{}), bool>::type = true>
inline cudaChannelFormatDesc get_channel_format_desc()
{
    return cudaCreateChannelDesc(sizeof(PixelType) * 8, 0, 0, 0, cudaChannelFormatKindSigned);
}

template <typename PixelType,
          typename std::enable_if<(!std::is_floating_point<PixelType>{} && std::is_unsigned<PixelType>{}), bool>::type = true>
inline cudaChannelFormatDesc get_channel_format_desc()
{
    return cudaCreateChannelDesc(sizeof(PixelType) * 8, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

class Layered2DTextureBuffer
{
private:
    // This is a special memory region **ONLY** for use with textures and surfaces.
    // It is non-addressable.
    // Addressable, linear memory uses different API calls and structs, centering around pitched ptrs.
    cudaArray_t array;
    // metadata for array
    cudaExtent extent;
    cudaChannelFormatDesc channel_desc;

    // Texture metadata and objects
    // Specify resource params for texture
    cudaResourceDesc tex_res_desc;
    // Specify texture object parameters
    cudaTextureDesc tex_desc;
    // texture object

public:
    cudaTextureObject_t tex_obj;

    template <typename PixelType,
              std::enable_if_t<std::is_arithmetic<PixelType>::value, bool> = true>
    Layered2DTextureBuffer(const ImageBuffer<PixelType> &buf,
                           bool use_normalized_coords,
                           bool use_normalized_float_read_mode = false,
                           cudaTextureFilterMode filter_mode = cudaFilterModeLinear,
                           cudaTextureAddressMode address_mode_0 = cudaAddressModeClamp,
                           cudaTextureAddressMode address_mode_1 = cudaAddressModeClamp)
        : extent(make_cudaExtent(buf.width, buf.height, buf.channels)),
          channel_desc(get_channel_format_desc<PixelType>())
    {
        checkCudaErrors(cudaMalloc3DArray(&array, &channel_desc, extent, cudaArrayLayered));

        // copy data to 3D array
        cudaMemcpy3DParms copy_params = {0};

        copy_params.srcPos = make_cudaPos(0, 0, 0);
        copy_params.dstPos = make_cudaPos(0, 0, 0);

        copy_params.dstArray = array;
        copy_params.srcPtr = make_cudaPitchedPtr(buf.data,
                                                 buf.pitch * sizeof(PixelType),
                                                 buf.width,
                                                 buf.height);
        copy_params.extent = extent;
        copy_params.kind = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&copy_params));

        memset(&tex_res_desc, 0, sizeof(cudaResourceDesc));
        tex_res_desc.resType = cudaResourceTypeArray;
        tex_res_desc.res.array.array = array;

        // clamp to boundaries for out of bounds indices
        memset(&tex_desc, 0, sizeof(cudaTextureDesc));
        tex_desc.addressMode[0] = address_mode_0;
        tex_desc.addressMode[1] = address_mode_1;
        // use bilinear interpolation for in-between indices
        tex_desc.filterMode = filter_mode;
        // the values read from the texture are the elements of the original array
        tex_desc.readMode =
            use_normalized_float_read_mode ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
        // normalize coordinates to [0, 1]
        tex_desc.normalizedCoords = use_normalized_coords;

        checkCudaErrors(cudaCreateTextureObject(&tex_obj, &tex_res_desc, &tex_desc, nullptr));
    }

    ~Layered2DTextureBuffer()
    {
        // Free up objects with allocated device memory
        checkCudaErrors(cudaDestroyTextureObject(tex_obj));
        checkCudaErrors(cudaFreeArray(array));
    }
};