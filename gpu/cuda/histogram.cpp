
#include <CImg.h>
#include <iostream>
#include <iomanip>
#include <string>

using cimg_library::CImg;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;
using std::string;

// Constants
const int HISTOGRAM_SIZE = 256;
const int BAR_WIDTH = 4;

extern void histogram1D(const int width, const int height, const unsigned char * inputImage, unsigned char * grayImage, unsigned int * histogram, unsigned char * histogramImage);


int main(int argc, char *argv[]) {
	unsigned int max = 0;

	if ( argc != 2 ) {
		cerr << "Usage: " << argv[0] << " <filename>" << endl;
		return 1;
	}

	// Load the input image
	CImg< unsigned char > inputImage = CImg< unsigned char >(argv[1]);
	if ( inputImage.spectrum() != 3 ) {
		cerr << "The input must be a color image." << endl;
		return 1;
	}

	// Convert the input image to grayscale and compute the histogram
	unsigned int * histogram = new unsigned int [HISTOGRAM_SIZE];
	CImg< unsigned char > grayImage = CImg< unsigned char >(inputImage.width(), inputImage.height(), 1, 1);
	CImg< unsigned char > histogramImage = CImg< unsigned char >(BAR_WIDTH * HISTOGRAM_SIZE, HISTOGRAM_SIZE, 1, 1);

	memset(reinterpret_cast< void * >(histogram), 0, HISTOGRAM_SIZE * sizeof(unsigned int));
	histogram1D(inputImage.width(), inputImage.height(), inputImage.data(), grayImage.data(), histogram, histogramImage.data());

	for ( int i = 0; i < HISTOGRAM_SIZE; i++ ) {
		if ( histogram[i] > max ) {
			max = histogram[i];
		}
	}

	for ( int x = 0; x < HISTOGRAM_SIZE * BAR_WIDTH; x += BAR_WIDTH ) {
		unsigned int value = HISTOGRAM_SIZE - ((histogram[x / BAR_WIDTH] * HISTOGRAM_SIZE) / max);

		for ( unsigned int y = 0; y < value; y++ ) {
			for ( int i = 0; i < BAR_WIDTH; i++ ) {
				histogramImage[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 0;
			}
		}
		for ( int y = value; y < HISTOGRAM_SIZE; y++ ) {
			for ( int i = 0; i < BAR_WIDTH; i++ ) {
				histogramImage[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 255;
			}
		}
	}

	// Save output
	grayImage.save(("./" + string(argv[1]) + ".gray.seq.bmp").c_str());
	histogramImage.save(("./" + string(argv[1]) + ".hist.seq.bmp").c_str());

	return 0;
}
