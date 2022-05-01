/*
    MIT License

    Copyright (c) 2022 HolyWu

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#include <algorithm>
#include <atomic>
#include <fstream>
#include <memory>
#include <semaphore>
#include <string>
#include <vector>

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include "waifu2x.h"

using namespace std::literals;

static std::atomic<int> numGPUInstances{ 0 };

struct Waifu2xData final {
    VSNode* node;
    VSVideoInfo vi;
    std::unique_ptr<Waifu2x> waifu2x;
    std::unique_ptr<std::counting_semaphore<>> semaphore;
};

static void filter(const VSFrame* src, VSFrame* dst, const Waifu2xData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept {
    const auto width{ vsapi->getFrameWidth(src, 0) };
    const auto height{ vsapi->getFrameHeight(src, 0) };
    const auto srcStride{ vsapi->getStride(src, 0) / d->vi.format.bytesPerSample };
    const auto dstStride{ vsapi->getStride(dst, 0) / d->vi.format.bytesPerSample };
    auto srcR{ reinterpret_cast<const float*>(vsapi->getReadPtr(src, 0)) };
    auto srcG{ reinterpret_cast<const float*>(vsapi->getReadPtr(src, 1)) };
    auto srcB{ reinterpret_cast<const float*>(vsapi->getReadPtr(src, 2)) };
    auto dstR{ reinterpret_cast<float*>(vsapi->getWritePtr(dst, 0)) };
    auto dstG{ reinterpret_cast<float*>(vsapi->getWritePtr(dst, 1)) };
    auto dstB{ reinterpret_cast<float*>(vsapi->getWritePtr(dst, 2)) };

    d->semaphore->acquire();
    d->waifu2x->process(srcR, srcG, srcB, dstR, dstG, dstB, width, height, srcStride, dstStride);
    d->semaphore->release();
}

static const VSFrame* VS_CC waifu2xGetFrame(int n, int activationReason, void* instanceData, [[maybe_unused]] void** frameData,
                                            VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<const Waifu2xData*>(instanceData) };

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        auto src{ vsapi->getFrameFilter(n, d->node, frameCtx) };
        auto dst{ vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core) };

        filter(src, dst, d, vsapi);

        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static void VS_CC waifu2xFree(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<Waifu2xData*>(instanceData) };
    vsapi->freeNode(d->node);
    delete d;

    if (--numGPUInstances == 0)
        ncnn::destroy_gpu_instance();
}

static void VS_CC waifu2xCreate(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData, VSCore* core, const VSAPI* vsapi) {
    auto d{ std::make_unique<Waifu2xData>() };

    try {
        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = *vsapi->getVideoInfo(d->node);
        int err;

        if (!vsh::isConstantVideoFormat(&d->vi) ||
            d->vi.format.colorFamily != cfRGB ||
            d->vi.format.sampleType != stFloat ||
            d->vi.format.bitsPerSample != 32)
            throw "only constant RGB format 32 bit float input supported";

        if (ncnn::create_gpu_instance())
            throw "failed to create GPU instance";
        ++numGPUInstances;

        auto noise{ vsapi->mapGetIntSaturated(in, "noise", 0, &err) };

        auto scale{ vsapi->mapGetIntSaturated(in, "scale", 0, &err) };
        if (err)
            scale = 2;

        auto tile_w{ vsapi->mapGetIntSaturated(in, "tile_w", 0, &err) };
        if (err)
            tile_w = std::max(d->vi.width, 32);

        auto tile_h{ vsapi->mapGetIntSaturated(in, "tile_h", 0, &err) };
        if (err)
            tile_h = std::max(d->vi.height, 32);

        auto model{ vsapi->mapGetIntSaturated(in, "model", 0, &err) };
        if (err)
            model = 2;

        auto gpuId{ vsapi->mapGetIntSaturated(in, "gpu_id", 0, &err) };
        if (err)
            gpuId = ncnn::get_default_gpu_index();

        auto gpuThread{ vsapi->mapGetIntSaturated(in, "gpu_thread", 0, &err) };
        if (err)
            gpuThread = 2;

        auto tta{ !!vsapi->mapGetInt(in, "tta", 0, &err) };
        auto fp32{ !!vsapi->mapGetInt(in, "fp32", 0, &err) };

        if (noise < -1 || noise > 3)
            throw "noise must be between -1 and 3 (inclusive)";

        if (scale < 1 || scale > 2)
            throw "scale must be 1 or 2";

        if (tile_w < 32)
            throw "tile_w must be at least 32";

        if (tile_h < 32)
            throw "tile_h must be at least 32";

        if (model < 0 || model > 2)
            throw "model must be between 0 and 2 (inclusive)";

        if (model != 2 && scale == 1)
            throw "only cunet model supports scale=1";

        if (gpuId < 0 || gpuId >= ncnn::get_gpu_count())
            throw "invalid GPU device";

        if (auto queue_count{ ncnn::get_gpu_info(gpuId).compute_queue_count() }; gpuThread < 1 || static_cast<uint32_t>(gpuThread) > queue_count)
            throw ("gpu_thread must be between 1 and " + std::to_string(queue_count) + " (inclusive)").c_str();

        if (!!vsapi->mapGetInt(in, "list_gpu", 0, &err)) {
            std::string text;

            for (auto i{ 0 }; i < ncnn::get_gpu_count(); i++)
                text += std::to_string(i) + ": " + ncnn::get_gpu_info(i).device_name() + "\n";

            auto args{ vsapi->createMap() };
            vsapi->mapConsumeNode(args, "clip", d->node, maReplace);
            vsapi->mapSetData(args, "text", text.c_str(), -1, dtUtf8, maReplace);

            auto ret{ vsapi->invoke(vsapi->getPluginByID(VSH_TEXT_PLUGIN_ID, core), "Text", args) };
            if (vsapi->mapGetError(ret)) {
                vsapi->mapSetError(out, vsapi->mapGetError(ret));
                vsapi->freeMap(args);
                vsapi->freeMap(ret);

                if (--numGPUInstances == 0)
                    ncnn::destroy_gpu_instance();

                return;
            }

            vsapi->mapConsumeNode(out, "clip", vsapi->mapGetNode(ret, "clip", 0, nullptr), maReplace);
            vsapi->freeMap(args);
            vsapi->freeMap(ret);

            if (--numGPUInstances == 0)
                ncnn::destroy_gpu_instance();

            return;
        }

        if (noise == -1 && scale == 1) {
            vsapi->mapConsumeNode(out, "clip", d->node, maReplace);

            if (--numGPUInstances == 0)
                ncnn::destroy_gpu_instance();

            return;
        }

        d->vi.width *= scale;
        d->vi.height *= scale;

        std::string pluginPath{ vsapi->getPluginPath(vsapi->getPluginByID("com.holywu.waifu2x-ncnn-Vulkan", core)) };
        auto modelDir{ pluginPath.substr(0, pluginPath.rfind('/')) + "/models" };

        int prepadding{};

        switch (model) {
        case 0:
            modelDir += "/models-upconv_7_anime_style_art_rgb";
            prepadding = 7;
            break;
        case 1:
            modelDir += "/models-upconv_7_photo";
            prepadding = 7;
            break;
        case 2:
            modelDir += "/models-cunet";
            prepadding = (noise == -1 || scale == 2) ? 18 : 28;
            break;
        }

        std::string paramPath, modelPath;

        if (noise == -1) {
            paramPath = modelDir + "/scale2.0x_model.param";
            modelPath = modelDir + "/scale2.0x_model.bin";
        } else if (scale == 1) {
            paramPath = modelDir + "/noise" + std::to_string(noise) + "_model.param";
            modelPath = modelDir + "/noise" + std::to_string(noise) + "_model.bin";
        } else {
            paramPath = modelDir + "/noise" + std::to_string(noise) + "_scale2.0x_model.param";
            modelPath = modelDir + "/noise" + std::to_string(noise) + "_scale2.0x_model.bin";
        }

        std::ifstream ifs{ paramPath };
        if (!ifs.is_open())
            throw "failed to load model";
        ifs.close();

        d->waifu2x = std::make_unique<Waifu2x>(gpuId, tta, 1);

#ifdef _WIN32
        auto paramBufferSize{ MultiByteToWideChar(CP_UTF8, 0, paramPath.c_str(), -1, nullptr, 0) };
        auto modelBufferSize{ MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, nullptr, 0) };
        std::vector<wchar_t> wparamPath(paramBufferSize);
        std::vector<wchar_t> wmodelPath(modelBufferSize);
        MultiByteToWideChar(CP_UTF8, 0, paramPath.c_str(), -1, wparamPath.data(), paramBufferSize);
        MultiByteToWideChar(CP_UTF8, 0, modelPath.c_str(), -1, wmodelPath.data(), modelBufferSize);
        d->waifu2x->load(wparamPath.data(), wmodelPath.data(), fp32);
#else
        d->waifu2x->load(paramPath, modelPath, fp32);
#endif

        d->waifu2x->noise = noise;
        d->waifu2x->scale = scale;
        d->waifu2x->tile_w = tile_w;
        d->waifu2x->tile_h = tile_h;
        d->waifu2x->prepadding = prepadding;

        d->semaphore = std::make_unique<std::counting_semaphore<>>(gpuThread);
    } catch (const char* error) {
        vsapi->mapSetError(out, ("waifu2x-ncnn-Vulkan: "s + error).c_str());
        vsapi->freeNode(d->node);

        if (--numGPUInstances == 0)
            ncnn::destroy_gpu_instance();

        return;
    }

    VSFilterDependency deps[]{ {d->node, rpStrictSpatial} };
    vsapi->createVideoFilter(out, "waifu2x-ncnn-Vulkan", &d->vi, waifu2xGetFrame, waifu2xFree, fmParallel, deps, 1, d.get(), core);
    d.release();
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.holywu.waifu2x-ncnn-Vulkan", "w2xncnnvk", "Image Super-Resolution using Deep Convolutional Neural Networks",
                         VS_MAKE_VERSION(2, 0), VAPOURSYNTH_API_VERSION, 0, plugin);

    vspapi->registerFunction("Waifu2x",
                             "clip:vnode;"
                             "noise:int:opt;"
                             "scale:int:opt;"
                             "tile_w:int:opt;"
                             "tile_h:int:opt;"
                             "model:int:opt;"
                             "gpu_id:int:opt;"
                             "gpu_thread:int:opt;"
                             "tta:int:opt;"
                             "fp32:int:opt;"
                             "list_gpu:int:opt;",
                             "clip:vnode;",
                             waifu2xCreate, nullptr, plugin);
}
