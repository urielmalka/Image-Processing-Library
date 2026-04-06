#include  "OCV.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>

#include <png.h>
#include <jpeglib.h>

OCV::OCV(){}

OCV::~OCV(){}

static std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return (char)std::tolower(c); });
    return s;
}

static std::string fileExtLower(const char* path) {
    if (!path) return "";
    std::string s(path);
    auto pos = s.find_last_of('.');
    if (pos == std::string::npos) return "";
    return toLower(s.substr(pos + 1));
}

static bool readFileAll(const char* path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    std::streamsize size = f.tellg();
    if (size <= 0) { out.clear(); return true; }
    f.seekg(0, std::ios::beg);
    out.resize((size_t)size);
    return (bool)f.read((char*)out.data(), size);
}

static bool decodeJPEG(const char* path, std::vector<uint8_t>& out, int& w, int& h, int& ch) {
    FILE* infile = std::fopen(path, "rb");
    if (!infile) return false;

    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);

    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&cinfo);
        std::fclose(infile);
        return false;
    }

    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);

    w = (int)cinfo.output_width;
    h = (int)cinfo.output_height;
    ch = (int)cinfo.output_components; // should be 3
    if (w <= 0 || h <= 0 || (ch != 3 && ch != 1)) {
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        std::fclose(infile);
        return false;
    }

    out.resize((size_t)w * (size_t)h * (size_t)ch);
    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW rowptr = (JSAMPROW)(out.data() + (size_t)cinfo.output_scanline * (size_t)w * (size_t)ch);
        jpeg_read_scanlines(&cinfo, &rowptr, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    std::fclose(infile);
    return true;
}

static bool decodePNG(const char* path, std::vector<uint8_t>& out, int& w, int& h, int& ch) {
    FILE* fp = std::fopen(path, "rb");
    if (!fp) return false;

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) { std::fclose(fp); return false; }
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_read_struct(&png, nullptr, nullptr); std::fclose(fp); return false; }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        std::fclose(fp);
        return false;
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    w = (int)png_get_image_width(png, info);
    h = (int)png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth  = png_get_bit_depth(png, info);

    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);

    // Normalize to either GRAY8 or RGB8 (drop alpha if present)
    png_read_update_info(png, info);

    png_byte updated_color_type = png_get_color_type(png, info);
    int rowbytes = (int)png_get_rowbytes(png, info);
    if (w <= 0 || h <= 0 || rowbytes <= 0) {
        png_destroy_read_struct(&png, &info, nullptr);
        std::fclose(fp);
        return false;
    }

    std::vector<uint8_t> raw((size_t)rowbytes * (size_t)h);
    std::vector<png_bytep> rows((size_t)h);
    for (int y = 0; y < h; y++) rows[(size_t)y] = (png_bytep)(raw.data() + (size_t)y * (size_t)rowbytes);
    png_read_image(png, rows.data());
    png_read_end(png, nullptr);

    png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(fp);

    // Determine channels in raw and convert.
    // Possible updated types include: GRAY, GRAY_ALPHA, RGB, RGBA
    int rawCh = 0;
    switch (updated_color_type) {
        case PNG_COLOR_TYPE_GRAY: rawCh = 1; break;
        case PNG_COLOR_TYPE_GRAY_ALPHA: rawCh = 2; break;
        case PNG_COLOR_TYPE_RGB: rawCh = 3; break;
        case PNG_COLOR_TYPE_RGBA: rawCh = 4; break;
        default: return false;
    }

    if (rawCh == 1) {
        ch = 1;
        out.resize((size_t)w * (size_t)h);
        for (int y = 0; y < h; y++) {
            std::memcpy(out.data() + (size_t)y * (size_t)w, raw.data() + (size_t)y * (size_t)rowbytes, (size_t)w);
        }
        return true;
    }

    // Convert everything else to RGB (drop alpha)
    ch = 3;
    out.resize((size_t)w * (size_t)h * 3u);
    for (int y = 0; y < h; y++) {
        const uint8_t* src = raw.data() + (size_t)y * (size_t)rowbytes;
        uint8_t* dst = out.data() + (size_t)y * (size_t)w * 3u;
        for (int x = 0; x < w; x++) {
            if (rawCh == 2) {
                uint8_t g = src[(size_t)x * 2u + 0u];
                dst[(size_t)x * 3u + 0u] = g;
                dst[(size_t)x * 3u + 1u] = g;
                dst[(size_t)x * 3u + 2u] = g;
            } else {
                dst[(size_t)x * 3u + 0u] = src[(size_t)x * (size_t)rawCh + 0u];
                dst[(size_t)x * 3u + 1u] = src[(size_t)x * (size_t)rawCh + 1u];
                dst[(size_t)x * 3u + 2u] = src[(size_t)x * (size_t)rawCh + 2u];
            }
        }
    }
    return true;
}

void OCV::ocvReadPixels(vector<vector<Pixel>> * data, int height, int width, int channels)
{
    if(channels == 1)
        data->resize(height, vector<Pixel>(width, Pixel{Grayscale{0}}));
    else
        data->resize(height, vector<Pixel>(width, Pixel{RGB{0,0,0}}));


    #pragma omp parallel for
    for (int  h= 0; h < height; h++) {

        if (channels == 1) {
            const unsigned char* rowGrayscale = decoded.data() + (size_t)h * (size_t)width;
            for (int w = 0; w < width ; w++) {
                (*data)[h][w] = Pixel{Grayscale{rowGrayscale[w]}};
            }
        } else {
            const unsigned char* rowRGB = decoded.data() + (size_t)h * (size_t)width * 3u;
            for (int w = 0; w < width; w++) {
                const unsigned char* px = rowRGB + (size_t)w * 3u;
                (*data)[h][w] = Pixel{RGB{px[0], px[1], px[2]}};
            }
        }
    }
}

void OCV::ocvSave(const char * path, const vector<vector<Pixel>> & data, int height, int width, bool isGrayscale)
{
    const int outCh = isGrayscale ? 1 : 3;
    std::vector<uint8_t> interleaved((size_t)width * (size_t)height * (size_t)outCh, 0);

    #pragma omp parallel for
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            const Pixel& px = data[h][w];
            if (outCh == 1) {
                uint8_t v = 0;
                if (px.type == PixelType::Grayscale) v = px.gray.I;
                else if (px.type == PixelType::RGB) v = (uint8_t)((int(px.rgb.R) + int(px.rgb.G) + int(px.rgb.B)) / 3);
                else if (px.type == PixelType::BGR) v = (uint8_t)((int(px.bgr.r) + int(px.bgr.g) + int(px.bgr.b)) / 3);
                interleaved[(size_t)h * (size_t)width + (size_t)w] = v;
            } else {
                uint8_t r=0,g=0,b=0;
                if (px.type == PixelType::RGB) { r = px.rgb.R; g = px.rgb.G; b = px.rgb.B; }
                else if (px.type == PixelType::BGR) { r = px.bgr.r; g = px.bgr.g; b = px.bgr.b; }
                else if (px.type == PixelType::Grayscale) { r = g = b = px.gray.I; }
                size_t idx = ((size_t)h * (size_t)width + (size_t)w) * 3u;
                interleaved[idx + 0u] = r;
                interleaved[idx + 1u] = g;
                interleaved[idx + 2u] = b;
            }
        }
    }

    const std::string ext = fileExtLower(path);
    if (ext == "png") {
        FILE* fp = std::fopen(path, "wb");
        if (!fp) { std::cerr << "Failed to open output file: " << path << std::endl; return; }

        png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
        if (!png) { std::fclose(fp); std::cerr << "Failed to create PNG writer\n"; return; }
        png_infop info = png_create_info_struct(png);
        if (!info) { png_destroy_write_struct(&png, nullptr); std::fclose(fp); std::cerr << "Failed to create PNG info\n"; return; }

        if (setjmp(png_jmpbuf(png))) {
            png_destroy_write_struct(&png, &info);
            std::fclose(fp);
            std::cerr << "Failed to write PNG\n";
            return;
        }

        png_init_io(png, fp);
        int colorType = (outCh == 1) ? PNG_COLOR_TYPE_GRAY : PNG_COLOR_TYPE_RGB;
        png_set_IHDR(png, info, (png_uint_32)width, (png_uint_32)height, 8, colorType,
                     PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
        png_write_info(png, info);

        std::vector<png_bytep> rows((size_t)height);
        const size_t stride = (size_t)width * (size_t)outCh;
        for (int y = 0; y < height; y++) rows[(size_t)y] = (png_bytep)(interleaved.data() + (size_t)y * stride);
        png_write_image(png, rows.data());
        png_write_end(png, nullptr);

        png_destroy_write_struct(&png, &info);
        std::fclose(fp);
        return;
    }

    // default: write JPEG (for .jpg/.jpeg or unknown)
    FILE* outfile = std::fopen(path, "wb");
    if (!outfile) { std::cerr << "Failed to open output file: " << path << std::endl; return; }

    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = (outCh == 1) ? 1 : 3;
    cinfo.in_color_space = (outCh == 1) ? JCS_GRAYSCALE : JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 90, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    const size_t rowStride = (size_t)width * (size_t)outCh;
    while (cinfo.next_scanline < cinfo.image_height) {
        JSAMPROW rowptr = (JSAMPROW)(interleaved.data() + (size_t)cinfo.next_scanline * rowStride);
        jpeg_write_scanlines(&cinfo, &rowptr, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    std::fclose(outfile);
}

void OCV::ocvSetWidthHeightChannels(int* height, int* width, int* channels)
{
    *height = decodedHeight;
    *width = decodedWidth;
    *channels = decodedChannels;
}

bool OCV::ocvOpenImage(const char *fn)
{
    decoded.clear();
    decodedWidth = decodedHeight = decodedChannels = 0;

    const std::string ext = fileExtLower(fn);
    bool ok = false;
    if (ext == "png") {
        ok = decodePNG(fn, decoded, decodedWidth, decodedHeight, decodedChannels);
    } else if (ext == "jpg" || ext == "jpeg") {
        ok = decodeJPEG(fn, decoded, decodedWidth, decodedHeight, decodedChannels);
    } else {
        // try JPEG then PNG as a fallback
        ok = decodeJPEG(fn, decoded, decodedWidth, decodedHeight, decodedChannels);
        if (!ok) ok = decodePNG(fn, decoded, decodedWidth, decodedHeight, decodedChannels);
    }

    if (!ok || decoded.empty() || decodedWidth <= 0 || decodedHeight <= 0 || (decodedChannels != 1 && decodedChannels != 3)) {
        std::cerr << "Error: Could not open/decode image: " << (fn ? fn : "(null)") << std::endl;
        decoded.clear();
        decodedWidth = decodedHeight = decodedChannels = 0;
        return false;
    }

    return true;
}