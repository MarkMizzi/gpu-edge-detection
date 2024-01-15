#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <vector>

template <typename PixelType,
          std::enable_if_t<std::is_arithmetic<PixelType>::value, bool> = true>
class ImageBuffer
{
public:
    const size_t width, height, channels;
    // NOTE: This pitch is in units of sizeof(PixelType), NOT in units of bytes like CUDA.
    const size_t pitch;

    PixelType *data;

    PixelType &operator()(size_t x, size_t y, size_t z)
    {
        return data[z * pitch * height + y * pitch + x];
    }

    const PixelType &operator()(size_t x, size_t y, size_t z) const
    {
        return data[z * pitch * height + y * pitch + x];
    }

    template <typename It>
    void read_from_row_ptrs(It begin, It end)
    {
        size_t y = 0;
        for (It it = begin; it != end; ++it, y++)
            for (size_t x = 0; x < width; x++)
                for (size_t z = 0; z < channels; z++)
                    (*this)(x, y, z) = (*it)[x * channels + z];
    }

    template <typename It>
    void write_to_row_ptrs(It begin, It end)
    {
        size_t y = 0;
        for (It it = begin; it != end; ++it, y++)
            for (size_t x = 0; x < width; x++)
                for (size_t z = 0; z < channels; z++)
                    (*it)[x * channels + z] = (*this)(x, y, z);
    }

    template <typename R,
              std::enable_if_t<std::is_arithmetic<R>::value, bool> = true>
    ImageBuffer<R> convert_to() const
    {
        ImageBuffer<R> converted(width, height, channels, pitch);

        for (size_t z = 0; z < channels; z++)
            for (size_t y = 0; y < height; y++)
                for (size_t x = 0; x < width; x++)
                    converted(x, y, z) = (R)(*this)(x, y, z);

        return converted;
    }

    void write_pgm(std::ostream &out)
    {
        static_assert(std::is_integral<PixelType>::value,
                      "Expected pixel type to be integral.");
        assert(channels == 1);

        out << "P5" << std::endl;
        out << width << std::endl;
        out << height << std::endl;
        out << static_cast<signed long long>(std::numeric_limits<PixelType>::max()) << std::endl;

        for (size_t r = 0; r < height; r++)
            out.write((char *)(data + pitch * r), width * sizeof(PixelType));
    }

    ImageBuffer(const size_t width, const size_t height, const size_t channels, size_t pitch = 0)
        : width(width),
          height(height),
          channels(channels),
          pitch(pitch == 0 ? width : pitch)
    {
        data = (PixelType *)calloc(channels * this->pitch * height, sizeof(PixelType));
    }

    ~ImageBuffer()
    {
        free(data);
    }

    ImageBuffer(const ImageBuffer &) = delete;
    ImageBuffer(ImageBuffer &&) = default;
};