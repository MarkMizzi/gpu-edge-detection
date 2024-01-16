# Canny Edge Detection in CUDA C++

This repository contains an implementation of Canny edge detection in CUDA C++.

Specifically, the pipeline
- Converts the image to grayscale (This is done on the host)
- Applies a Gaussian blur
- Applies the Sobel gradient kernels to determine the magnitude and direction of the gradient at different points of the image.
- Uses maximum suppression technique to filter out noise from the gradient magnitude image.
- Uses double thresholding (upper and lower thresholds) to filter out more noise. 

## Examples

```{sh}
./target/edging -i <input_filename> -o dome.png -l 0.53
```

Input:
![IMG_2998-min](https://github.com/MarkMizzi/gpu-edge-detection/assets/88614379/0b006f7d-e676-45f2-8414-f27839d1f415)
Output:
![dome](https://github.com/MarkMizzi/gpu-edge-detection/assets/88614379/27ee148d-94f8-4149-9bc6-31abd002c953)

## Building

To build the program, use the following commands:
```{sh}
# for the release build
make
# for the debug build
make dbg=1
# build for profiling
make prof=1
```

This produces a binary at `target/edging`. The binary and object files from the build can be removed by running
```{bash}
make clean
```

## Usage

The program only supports RGB PNG images (no alpha channel) at the moment. 
If you have other input, please convert it using a program such as Gimp before using it as input to the program.

To see the available options, run
```{sh}
./target/edging --help
```

This outputs the following (may not be up to date):
```{text}
Usage:
  -h [ --help ]                Issue help message and exit.
  -s [ --blur_stddev ] arg     Standard deviation of Gaussian blur used 
                               (default 1).
  -x [ --blur_rad_x ] arg      x-radius of Gaussian blur used. (default 1).
  -y [ --blur_rad_y ] arg      y-radius of Gaussian blur used. (default 1).
  -u [ --upper_threshold ] arg Upper threshold to apply to image with edges 
                               [0-1]. (default 1).
  -l [ --lower_threshold ] arg Lower threshold to apply to image with edges 
                               [0-1]. (default 0.5).
  -i [ --input ] arg           Path to input image. (Required).
  -o [ --output ] arg          Path to which output should be written. 
                               (Required).
```
