#include "edge_detection.cuh"
#include <png.hh>

#include <boost/program_options.hpp>

#include <limits>
#include <cassert>
#include <fstream>
#include <ios>
#include <iostream>
#include <stdexcept>
#include <string>

extern "C"
{
#include <sys/time.h>
}

namespace po = boost::program_options;

#define CHECK_REQUIRED_ARG(DESC, VM, ARG)                                      \
    if (!(VM).count(ARG))                                                      \
    {                                                                          \
        std::cout << "--" << (ARG) << " is a required argument." << std::endl; \
        std::cout << (DESC) << std::endl;                                      \
    }

template <typename T>
inline T get_optional_arg(po::variables_map &vm,
                          const std::string &optname,
                          const T default_val)
{
    T val = default_val;

    if (vm.count(optname))
        val = vm[optname].as<T>();

    return val;
}

int main(int argc, char *argv[])
{

    // Declare the supported options.
    po::options_description desc("Usage");
    desc.add_options()("help,h", "Issue help message and exit.")(
        "blur_stddev,s", po::value<double>(),
        "Standard deviation of Gaussian blur used (default 1).")(
        "blur_rad_x,x", po::value<ssize_t>(),
        "x-radius of Gaussian blur used. (default 1).")(
        "blur_rad_y,y", po::value<ssize_t>(),
        "y-radius of Gaussian blur used. (default 1).")(
        "upper_threshold,u", po::value<double>(),
        "Upper threshold to apply to image with edges [0-1]. (default 1).")(
        "lower_threshold,l", po::value<double>(),
        "Lower threshold to apply to image with edges [0-1]. (default 0.5).")(
        "input,i", po::value<std::string>(),
        "Path to input image. (Required).")(
        "output,o", po::value<std::string>(),
        "Path to which output should be written. (Required).");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 1;
    }

    CHECK_REQUIRED_ARG(desc, vm, "input");
    CHECK_REQUIRED_ARG(desc, vm, "output");

    // size of neighbourhood used when computing disparity
    ssize_t blur_rad_x = get_optional_arg(vm, "blur_rad_x", 1);
    ssize_t blur_rad_y = get_optional_arg(vm, "blur_rad_y", 1);

    double blur_stddev = get_optional_arg(vm, "blur_stddev", 1);

    double upper_threshold = get_optional_arg(vm, "upper_threshold", 1);
    double lower_threshold = get_optional_arg(vm, "lower_threshold", 0.5);

    std::string input_fname = vm["input"].as<std::string>();
    std::string output_fname = vm["output"].as<std::string>();

    std::fstream input_png_file(input_fname, std::ios::binary | std::ios::in);

    PNGReader reader;

    ImageBuffer<PNGReader::PixelType> input_img = reader.read(input_png_file);

    // Record start of algorithm
    struct timeval start;
    gettimeofday(&start, nullptr);

    ImageBuffer<PNGWriter::PixelType> edges_image =
        edge_detect(input_img, blur_stddev, blur_rad_x, blur_rad_y, upper_threshold, lower_threshold);

    // Record end of algorithm
    struct timeval end;
    gettimeofday(&end, nullptr);

    // Output duration
    suseconds_t duration = (end.tv_sec - start.tv_sec) * 1e6 + end.tv_usec - start.tv_usec;
    std::cout << duration << std::endl;

    std::fstream png_outfile(output_fname, std::ios::binary | std::ios::out);

    PNGWriter writer;
    writer.write(png_outfile, edges_image, PNG_COLOR_TYPE_GRAY);

    return 0;
}