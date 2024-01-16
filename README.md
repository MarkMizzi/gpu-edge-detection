# Canny edge detector in CUDA C++

This repository contains an implementation of Canny edge detection in CUDA C++.

Specifically, the pipeline
- Converts the image to grayscale (This is done on the host)
- Applies a Gaussian blur
- Applies the Sobel gradient kernels to determine the magnitude and direction of the gradient at different points of the image.
- Uses maximum suppression technique to filter out noise from the gradient magnitude image.
- Uses double thresholding (upper and lower thresholds) to filter out more noise. 

## Examples

```{sh}
./target/edging -i dome.png -o dome-edges.png -l 0.53
```

Input:
![IMG_2998-min](https://github.com/MarkMizzi/gpu-edge-detection/assets/88614379/0b006f7d-e676-45f2-8414-f27839d1f415)
Output:
![dome](https://github.com/MarkMizzi/gpu-edge-detection/assets/88614379/27ee148d-94f8-4149-9bc6-31abd002c953)

```{sh}
./target/edging -i jaguar.png -o jaguar-edges.png -x 2 -y 2 -s 2.5 -l 0.53
```

Input:
![jaguar-min](https://github.com/MarkMizzi/gpu-edge-detection/assets/88614379/9123fd20-7bcb-4f52-a0bb-b920c727d5a7)
Output:
![jaguar-edges](https://github.com/MarkMizzi/gpu-edge-detection/assets/88614379/3b46a240-513e-43ad-aba8-58e122a33324)

```{sh}
./target/edging -i dome-of-the-rock.png -o dome-of-the-rock-edges.png -x 1 -y 1 -s 2.0 -l 0.52
```

Input:
![dome-of-the-rock-min](https://github.com/MarkMizzi/gpu-edge-detection/assets/88614379/9afa7ca6-630a-4a24-be03-e86d315cfa5c)
Output:
![dome-of-the-rock-edges](https://github.com/MarkMizzi/gpu-edge-detection/assets/88614379/59a9696a-d152-4382-8995-d8e458f1abf2)

```{sh}
./target/edging -i astronaut.png -o astronaut-edges.png -x 2 -y 2 -s 2.0 -l 0.53
```

Input:
![astronaut-min](https://github.com/MarkMizzi/gpu-edge-detection/assets/88614379/96fe513e-0f56-47ce-b524-616ce1e978d7)
Output:
![astronaut-edges](https://github.com/MarkMizzi/gpu-edge-detection/assets/88614379/1875f905-6122-4c97-82ef-502e689c13d4)

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

The program does not support indexed PNG images as input at the moment. 
If you have such an input, please convert it to RGB or grayscale using a program like Gimp before using it as input to the program.

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
