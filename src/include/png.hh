#pragma once

#include "imagebuf.hh"

extern "C"
{
#include <png.h>
}

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

class PNGReadError : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

class PNGReader
{
public:
    using PixelType = uint8_t;
    static constexpr int png_sig_size = 8;

private:
    png_structp png_handle = nullptr;
    png_infop png_info = nullptr;

    png_uint_32 width = 0, height = 0, channels = 0, color_type = 0;

    std::unique_ptr<ImageBuffer<PixelType>> image_data = nullptr;

    static bool validate(std::istream &source)
    {
        png_byte sig[png_sig_size];

        source.read((char *)sig, png_sig_size);

        // check if read from stream succeeded.
        if (!source.good())
            return false;

        return png_sig_cmp(sig, 0, png_sig_size) == 0;
    }

public:
    ImageBuffer<PixelType> read(std::istream &source)
    {
        /// validate that source contains a PNG file
        if (!validate(source))
            throw PNGReadError("The given stream is not a valid PNG file.");

        /// Create the structs necessary for reading the PNG

        png_handle = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr,
                                            nullptr);
        if (!png_handle)
            throw PNGReadError("Failed to create a new PNG read struct");

        png_info = png_create_info_struct(png_handle);
        if (!png_info)
            throw PNGReadError("Failed to create a new PNG info struct");

        /// Setup png read function

        png_set_read_fn(
            png_handle, (png_voidp)&source,
            [](png_structp png_handle, png_bytep data, png_size_t length)
            {
                png_voidp ioptr = png_get_io_ptr(png_handle);
                ((std::istream *)ioptr)->read((char *)data, length);
            });

        /// read the PNG header

        // skip the PNG signature we have already read
        png_set_sig_bytes(png_handle, png_sig_size);

        // read in the png header
        png_read_info(png_handle, png_info);

        // fill in parameters of the image.
        width = png_get_image_width(png_handle, png_info);
        height = png_get_image_height(png_handle, png_info);

        channels = png_get_channels(png_handle, png_info);

        color_type = png_get_color_type(png_handle, png_info);

        // normalize to RGB format
        if (color_type == PNG_COLOR_TYPE_PALETTE)
            png_set_palette_to_rgb(png_handle);
        if (color_type & PNG_COLOR_MASK_ALPHA)
            png_set_strip_alpha(png_handle);

        /// read image data
        ImageBuffer<PixelType> buf(width, height, channels);

        // temporary buffer to load things in row-major order.
        std::vector<PixelType *> row_ptrs(height);

        png_byte bit_depth = png_get_bit_depth(png_handle, png_info);

        // normalize to 8-bit pixel values
        if (bit_depth < 8)
            png_set_packing(png_handle);
        else if (bit_depth == 16)
            png_set_strip_16(png_handle);

        // allocate buffer for each row.
        size_t bufsize = png_get_rowbytes(png_handle, png_info);
        for (auto it = row_ptrs.begin(); it != row_ptrs.end(); ++it)
            *it = (PixelType *)calloc(bufsize, sizeof(char));

        png_read_image(png_handle, (png_bytep *)row_ptrs.data());

        // copy pixels into ImageBuffer reorganizing the memory in the process.
        buf.read_from_row_ptrs(row_ptrs.begin(), row_ptrs.end());

        for (auto it = row_ptrs.begin(); it != row_ptrs.end(); ++it)
            free(*it);

        return buf;
    }

    PNGReader() {}

    ~PNGReader()
    {
        if (png_handle)
            png_destroy_read_struct(&png_handle, &png_info, (png_infopp)0);
    }
};

class PNGWriteError : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

class PNGWriter
{
public:
    using PixelType = uint8_t;
    static constexpr int png_sig_size = 8;

private:
    png_structp png_handle = nullptr;
    png_infop png_info = nullptr;

public:
    void write(std::ostream &output, ImageBuffer<PixelType> &buf, png_byte color_type = PNG_COLOR_TYPE_RGB)
    {
        png_handle = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

        if (!png_handle)
            throw PNGReadError("Failed to create a new PNG write struct");

        png_info = png_create_info_struct(png_handle);
        if (!png_info)
            throw PNGReadError("Failed to create a new PNG info struct");

        if (setjmp(png_jmpbuf(png_handle)))
            throw PNGReadError("setjmp failed.");

        png_set_write_fn(
            png_handle, (png_voidp)&output,
            [](png_structp png_handle, png_bytep data, png_size_t length)
            {
                png_voidp ioptr = png_get_io_ptr(png_handle);
                ((std::ostream *)ioptr)->write((char *)data, length);
            },
            nullptr);

        // Output is 16bit depth, RGB format.
        png_set_IHDR(
            png_handle,
            png_info,
            buf.width, buf.height,
            sizeof(PixelType) * 8,
            color_type,
            PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_DEFAULT,
            PNG_FILTER_TYPE_DEFAULT);
        png_write_info(png_handle, png_info);

        // To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
        // Use png_set_filler().
        // png_set_filler(png, 0, PNG_FILLER_AFTER);

        std::vector<PixelType *> row_ptrs(buf.height);

        for (auto it = row_ptrs.begin(); it != row_ptrs.end(); ++it)
            *it = (PixelType *)calloc(
                buf.width * buf.channels, sizeof(PixelType));

        buf.write_to_row_ptrs(row_ptrs.begin(), row_ptrs.end());

        png_write_image(png_handle, (png_bytep *)row_ptrs.data());
        png_write_end(png_handle, nullptr);

        for (auto it = row_ptrs.begin(); it != row_ptrs.end(); ++it)
            free(*it);
    }

    PNGWriter() {}

    ~PNGWriter()
    {
        if (png_handle)
            // Free write struct
            png_destroy_write_struct(&png_handle, &png_info);
    }
};