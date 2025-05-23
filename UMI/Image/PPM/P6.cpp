#include "P6.hpp"

PPM_P6::PPM_P6(const char* fn): ImagePPM(fn, P6)
{
    channels = P6_CHANNEL_NUM;

    readPixels();
    loadFileSuccess = true;
};

PPM_P6::~PPM_P6() {};

void PPM_P6::readPixels()
{

    data.resize(height, vector<Pixels>(width, RGB{0,0,0}));

    string line;
    istringstream iss(line);
    unsigned char r,g,b;

    for(int h=0; h < height; h++){
        for(int w=0; w < width; w++){
            image.read(reinterpret_cast<char *>(&r), 1);
            image.read(reinterpret_cast<char *>(&g), 1);
            image.read(reinterpret_cast<char *>(&b), 1);

            iss >> r >> g >> b;

            data[h][w] = RGB{r,g,b};
        }
    }
    
    image.close();
};

void PPM_P6::save(const char* path)
{

    ofstream out(path, ios::binary);
    if (!out.is_open()) {
        cerr << "Failed to open file for writing: " << filename << endl;
        return;
    }

    // Write P6 header
    out << "P6\n" << width << " " << height << "\n255\n";

    vector<RGB> tempFlatData;
    tempFlatData.reserve(height * width * 3);

    // Write binary pixel data
    for (int h = 0; h < height; ++h) {
        for (auto& pixel : data[h]) {
            if (std::holds_alternative<RGB>(pixel)) {
                tempFlatData.push_back(std::get<RGB>(pixel));
            }
            else {
                cerr << "Non-RGB pixel found in data!" << endl;
                // Optional: convert BGR/Grayscale/BinPixel to RGB here if needed
            }
        }
    }

    out.write(reinterpret_cast<const char*>(tempFlatData.data()), width * height * 3);

    out.close();
}
