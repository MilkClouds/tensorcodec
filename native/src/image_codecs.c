/* Dedicated image codecs. All longjmp error handling stays within C. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <setjmp.h>
#include <jpeglib.h>
#include <webp/decode.h>
#include <webp/demux.h>
#include <avif/avif.h>
#include "gif_lib.h"

typedef struct {
    void *data;
    size_t frames, height, width, channels, bytes;
    int bits, orientation, animated;
    char error[256];
} Output;

static int fail(Output *out, const char *message) {
    if (message != out->error) snprintf(out->error, sizeof(out->error), "%s", message);
    return 0;
}

static int allocate(Output *out, size_t n, size_t h, size_t w, size_t c, int bits) {
    size_t size = bits / 8;
    const size_t dims[] = {n, h, w, c};
    for (size_t i = 0; i < 4; ++i) {
        if (!dims[i] || dims[i] > (size_t)PTRDIFF_MAX / size)
            return fail(out, "invalid or overflowing image dimensions");
        size *= dims[i];
    }
    out->data = calloc(1, size);
    if (!out->data) return fail(out, "cannot allocate image output");
    out->frames = n; out->height = h; out->width = w; out->channels = c;
    out->bits = bits; out->bytes = size;
    return 1;
}

typedef struct {
    struct jpeg_decompress_struct decoder;
    struct jpeg_error_mgr error;
    jmp_buf jump;
    Output *out;
    uint8_t *row;
} Jpeg;

static void jpeg_error(j_common_ptr common) {
    Jpeg *ctx = common->client_data;
    char message[JMSG_LENGTH_MAX];
    common->err->format_message(common, message);
    fail(ctx->out, message);
    longjmp(ctx->jump, 1);
}
static void jpeg_message(j_common_ptr common, int level) {
    if (level < 0) jpeg_error(common); /* Never synthesize missing EOI data. */
}
static int decode_jpeg(const uint8_t *data, size_t size, int mode, Output *out) {
    Jpeg *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return fail(out, "cannot allocate JPEG context");
    ctx->out = out;
    ctx->decoder.err = jpeg_std_error(&ctx->error);
    ctx->error.error_exit = jpeg_error;
    ctx->error.emit_message = jpeg_message;
    ctx->decoder.client_data = ctx;
    if (setjmp(ctx->jump)) {
        jpeg_destroy_decompress(&ctx->decoder);
        free(ctx->row); free(ctx);
        return 0;
    }
    jpeg_create_decompress(&ctx->decoder);
    jpeg_mem_src(&ctx->decoder, data, size);
    jpeg_read_header(&ctx->decoder, TRUE);
    int cmyk = ctx->decoder.jpeg_color_space == JCS_CMYK || ctx->decoder.jpeg_color_space == JCS_YCCK;
    int gray = mode == 1 || mode == 2;
    if (!cmyk && mode) ctx->decoder.out_color_space = gray ? JCS_GRAYSCALE : JCS_RGB;
    jpeg_start_decompress(&ctx->decoder);
    size_t channels = cmyk && mode ? (gray ? 1 : 3) : (size_t)ctx->decoder.output_components;
    if (!allocate(out, 1, ctx->decoder.output_height, ctx->decoder.output_width, channels, 8)) {
        jpeg_destroy_decompress(&ctx->decoder); free(ctx); return 0;
    }
    if (cmyk && mode) {
        ctx->row = malloc(out->width * 4);
        if (!ctx->row) {
            jpeg_destroy_decompress(&ctx->decoder); free(ctx);
            return fail(out, "cannot allocate CMYK scanline");
        }
    }
    while (ctx->decoder.output_scanline < ctx->decoder.output_height) {
        uint8_t *dest = (uint8_t *)out->data + ctx->decoder.output_scanline * out->width * channels;
        uint8_t *row = ctx->row ? ctx->row : dest;
        jpeg_read_scanlines(&ctx->decoder, &row, 1);
        if (ctx->row) {
            for (size_t x = 0; x < out->width; ++x) {
                int rgb[3], k = row[4*x+3];
                for (int c = 0; c < 3; ++c) {
                    int v = k * (255 - row[4*x+c]) + 128;
                    rgb[c] = k - (((v >> 8) + v) >> 8);
                }
                if (gray) dest[x] = (rgb[0]*19595 + rgb[1]*38470 + rgb[2]*7471 + 32768) >> 16;
                else for (int c = 0; c < 3; ++c) dest[3*x+c] = rgb[c];
            }
        } else if (cmyk) {
            for (size_t x = 0; x < out->width * 4; ++x) dest[x] = 255 - dest[x];
        }
    }
    jpeg_finish_decompress(&ctx->decoder);
    jpeg_destroy_decompress(&ctx->decoder);
    free(ctx->row); free(ctx);
    return 1;
}

typedef struct { const uint8_t *data; size_t left; Output *out; } Source;
static int decode_webp(const uint8_t *data, size_t size, int mode, Output *out) {
    WebPBitstreamFeatures features;
    if (WebPGetFeatures(data, size, &features) != VP8_STATUS_OK) return fail(out, "invalid WebP header");
    int channels = mode == 4 || mode == 2 || (!mode && features.has_alpha) ? 4 : 3;
    if (!features.has_animation) {
        if (!allocate(out, 1, features.height, features.width, channels, 8)) return 0;
        if (out->width > INT_MAX / (size_t)channels) return fail(out, "WebP stride overflow");
        uint8_t *result = channels == 3
            ? WebPDecodeRGBInto(data, size, out->data, out->bytes, out->width * channels)
            : WebPDecodeRGBAInto(data, size, out->data, out->bytes, out->width * channels);
        return result ? 1 : fail(out, "invalid or truncated WebP");
    }
    WebPData source = {data, size};
    WebPAnimDecoderOptions options;
    if (!WebPAnimDecoderOptionsInit(&options)) return fail(out, "incompatible WebP animation ABI");
    options.color_mode = MODE_RGBA;
    WebPAnimDecoder *decoder = WebPAnimDecoderNew(&source, &options);
    if (!decoder) return fail(out, "invalid WebP animation");
    WebPAnimInfo info;
    int ok = WebPAnimDecoderGetInfo(decoder, &info);
    if (!ok || !allocate(out, info.frame_count, info.canvas_height, info.canvas_width, channels, 8)) {
        WebPAnimDecoderDelete(decoder); return fail(out, "invalid WebP animation dimensions");
    }
    out->animated = 1;
    size_t i = 0, pixels = out->width * out->height;
    while (WebPAnimDecoderHasMoreFrames(decoder)) {
        uint8_t *frame; int timestamp;
        if (i >= out->frames || !WebPAnimDecoderGetNext(decoder, &frame, &timestamp)) { ok = 0; break; }
        uint8_t *dest = (uint8_t *)out->data + i++ * pixels * channels;
        if (channels == 4) memcpy(dest, frame, pixels * 4);
        else for (size_t j = 0; j < pixels; ++j) memcpy(dest + j*3, frame + j*4, 3);
    }
    WebPAnimDecoderDelete(decoder);
    return ok && i == out->frames ? 1 : fail(out, "incomplete WebP animation");
}

static int decode_avif(const uint8_t *data, size_t size, int mode, int depth, int threads, Output *out) {
    avifDecoder *decoder = avifDecoderCreate();
    if (!decoder) return fail(out, "cannot allocate AVIF decoder");
    decoder->maxThreads = threads;
    avifResult result = avifDecoderSetIOMemory(decoder, data, size);
    if (result == AVIF_RESULT_OK) result = avifDecoderParse(decoder);
    if (result != AVIF_RESULT_OK) goto error;
    avifImage *image = decoder->image;
    int angle = image->transformFlags & AVIF_TRANSFORM_IROT ? image->irot.angle : 0;
    int mirror = image->transformFlags & AVIF_TRANSFORM_IMIR;
    const int rotations[] = {1, 8, 3, 6}, mirror0[] = {4, 5, 2, 7}, mirror1[] = {2, 7, 4, 5};
    out->orientation = mirror ? (image->imir.axis ? mirror1[angle & 3] : mirror0[angle & 3]) : rotations[angle & 3];
    size_t n = decoder->imageSequenceTrackPresent ? decoder->imageCount : 1;
    int channels = mode == 4 || mode == 2 || (!mode && decoder->alphaPresent) ? 4 : 3;
    int high = depth == 16 || (!depth && image->depth > 8);
    if (!allocate(out, n, image->height, image->width, channels, high ? 16 : 8)) { avifDecoderDestroy(decoder); return 0; }
    for (size_t i = 0; i < n; ++i) {
        result = avifDecoderNextImage(decoder);
        if (result != AVIF_RESULT_OK) goto error;
        if (image->width != out->width || image->height != out->height) {
            avifDecoderDestroy(decoder); return fail(out, "AVIF frames have different dimensions");
        }
        avifRGBImage rgb;
        avifRGBImageSetDefaults(&rgb, image);
        rgb.depth = high ? 16 : 8;
        rgb.format = channels == 3 ? AVIF_RGB_FORMAT_RGB : AVIF_RGB_FORMAT_RGBA;
        rgb.ignoreAlpha = channels == 3;
        rgb.pixels = (uint8_t *)out->data + i * out->bytes / n;
        size_t stride = out->width * channels * (high ? 2 : 1);
        if (stride > UINT32_MAX) { avifDecoderDestroy(decoder); return fail(out, "AVIF stride overflow"); }
        rgb.rowBytes = (uint32_t)stride;
        result = avifImageYUVToRGB(image, &rgb);
        if (result != AVIF_RESULT_OK) goto error;
    }
    avifDecoderDestroy(decoder); return 1;
error:
    fail(out, avifResultToString(result)); avifDecoderDestroy(decoder); return 0;
}

static int gif_read(GifFileType *gif, GifByteType *dest, int size) {
    Source *source = gif->UserData;
    if (size < 0) return 0;
    size_t n = (size_t)size < source->left ? (size_t)size : source->left;
    memcpy(dest, source->data, n); source->data += n; source->left -= n;
    return (int)n;
}
static void gif_clear(uint8_t *canvas, size_t h, size_t w, int c, GifColorType bg, GifImageDesc rect) {
    for (int y = 0; y < rect.Height; ++y) {
        size_t row = (size_t)rect.Top + y;
        if (row >= h) continue;
        for (int x = 0; x < rect.Width; ++x) {
            size_t col = (size_t)rect.Left + x;
            if (col >= w) continue;
            uint8_t *p = canvas + (row*w + col)*c;
            if (c == 4) memset(p, 0, 4);
            else { p[0] = bg.Red; p[1] = bg.Green; p[2] = bg.Blue; }
        }
    }
}
static int decode_gif(const uint8_t *data, size_t size, int mode, Output *out) {
    Source source = {data, size, out};
    int error = 0, ok = 0;
    GifFileType *gif = DGifOpen(&source, gif_read, &error);
    if (!gif) return fail(out, "invalid GIF header");
    uint8_t *restore = NULL;
    if (DGifSlurp(gif) == GIF_ERROR || gif->ImageCount <= 0) { fail(out, "invalid or truncated GIF"); goto done; }
    int alpha = 0;
    for (int i = 0; i < gif->ImageCount; ++i) {
        GraphicsControlBlock control = {0, false, 0, NO_TRANSPARENT_COLOR};
        DGifSavedExtensionToGCB(gif, i, &control);
        alpha |= control.TransparentColor != NO_TRANSPARENT_COLOR;
    }
    int c = mode == 4 || mode == 2 || (!mode && alpha) ? 4 : 3;
    size_t h = gif->SHeight, w = gif->SWidth;
    if ((size_t)gif->SavedImages[0].ImageDesc.Height > h) h = gif->SavedImages[0].ImageDesc.Height;
    if ((size_t)gif->SavedImages[0].ImageDesc.Width > w) w = gif->SavedImages[0].ImageDesc.Width;
    if (!allocate(out, gif->ImageCount, h, w, c, 8)) goto done;
    size_t frame_bytes = out->bytes / out->frames;
    GifColorType bg = {0, 0, 0};
    if (gif->SColorMap) {
        if (gif->SBackGroundColor < 0 || gif->SBackGroundColor >= gif->SColorMap->ColorCount) { fail(out, "invalid GIF background"); goto done; }
        bg = gif->SColorMap->Colors[gif->SBackGroundColor];
    }
    int disposal = 0;
    GifImageDesc previous = {0};
    for (int i = 0; i < gif->ImageCount; ++i) {
        uint8_t *canvas = (uint8_t *)out->data + i * frame_bytes;
        GraphicsControlBlock control = {0, false, 0, NO_TRANSPARENT_COLOR};
        DGifSavedExtensionToGCB(gif, i, &control);
        if (!i) {
            GifImageDesc whole = {0}; whole.Width = w; whole.Height = h;
            gif_clear(canvas, h, w, c, bg, whole);
        } else if (disposal == DISPOSE_PREVIOUS && restore) memcpy(canvas, restore, frame_bytes);
        else {
            memcpy(canvas, canvas - frame_bytes, frame_bytes);
            if (disposal == DISPOSE_BACKGROUND) gif_clear(canvas, h, w, c, bg, previous);
        }
        if (control.DisposalMode == DISPOSE_PREVIOUS) {
            if (!restore) restore = malloc(frame_bytes);
            if (!restore) { fail(out, "cannot allocate GIF restore canvas"); goto done; }
            memcpy(restore, canvas, frame_bytes);
        }
        SavedImage *image = &gif->SavedImages[i];
        GifImageDesc desc = image->ImageDesc;
        ColorMapObject *map = desc.ColorMap ? desc.ColorMap : gif->SColorMap;
        if (!map) { fail(out, "GIF has no color map"); goto done; }
        for (int y = 0; y < desc.Height; ++y) for (int x = 0; x < desc.Width; ++x) {
            int index = image->RasterBits[(size_t)y*desc.Width+x];
            if (index == control.TransparentColor) continue;
            if (index >= map->ColorCount) { fail(out, "invalid GIF palette index"); goto done; }
            size_t row = (size_t)desc.Top+y, col = (size_t)desc.Left+x;
            if (row >= h || col >= w) continue;
            uint8_t *p = canvas + (row*w+col)*c;
            p[0] = map->Colors[index].Red; p[1] = map->Colors[index].Green; p[2] = map->Colors[index].Blue;
            if (c == 4) p[3] = 255;
        }
        disposal = control.DisposalMode; previous = desc;
    }
    ok = 1;
done:
    free(restore); DGifCloseFile(gif, &error); return ok;
}

int tc_decode_image(const uint8_t *data, size_t size, int codec, int mode, int depth, int threads, Output *out) {
    out->orientation = 1;
    switch (codec) {
        case 0: return decode_jpeg(data, size, mode, out);
        case 2: return decode_webp(data, size, mode, out);
        case 3: return decode_gif(data, size, mode, out);
        case 4: return decode_avif(data, size, mode, depth, threads, out);
        default: return fail(out, "unknown image codec");
    }
}
