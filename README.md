# waifu2x ncnn Vulkan
[![CI](https://github.com/HolyWu/vs-waifu2x-ncnn-vulkan/actions/workflows/CI.yml/badge.svg)](https://github.com/HolyWu/vs-waifu2x-ncnn-vulkan/actions/workflows/CI.yml)

ncnn implementation of waifu2x converter, based on [waifu2x-ncnn-vulkan](https://github.com/nihui/waifu2x-ncnn-vulkan).


## Usage
    w2xncnnvk.Waifu2x(vnode clip[, int noise=0, int scale=2, int tile=0, int model=2, int gpu_id=None, int gpu_thread=2, bint tta=False, bint fp32=False, bint list_gpu=False])

- clip: Clip to process. Only RGB format with float sample type of 32 bit depth is supported.

- noise: Denoise level (-1/0/1/2/3). Large value means strong denoise effect, -1 = no effect.

- scale: Upscale ratio (1/2).

- tile: Tile size (>=32/0=auto). Use smaller value to reduce GPU memory usage. 

- model: Model to use.
  - 0 = upconv_7_anime_style_art_rgb
  - 1 = upconv_7_photo
  - 2 = cunet

- gpu_id: GPU device to use.

- gpu_thread: Thread count for upscaling. Using larger values may increase GPU usage and consume more GPU memory. If you find that your GPU is hungry, try increasing thread count to achieve faster processing.

- tta: Enable TTA(Test-Time Augmentation) mode.

- fp32: Enable FP32 mode.

- list_gpu: Simply print a list of available GPU devices on the frame and does nothing else.


## Compilation
Requires `Vulkan SDK`.

```
git submodule update --init --recursive --depth 1
meson build
ninja -C build
ninja -C build install
```
