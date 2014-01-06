
#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;


void histogram1D(const int width, const int height, const unsigned char * inputImage, unsigned char * grayImage, unsigned int * histogram, unsigned char * histogramImage) {
	NSTimer kernelTime = NSTimer("histogram", false, false);
	
	kernelTime.start();
	// Kernel
	for ( int y = 0; y < height; y++ ) {
		for ( int x = 0; x < width; x++ ) {
			float grayPix = 0.0f;
			float r = static_cast< float >(inputImage[(y * width) + x]);
			float g = static_cast< float >(inputImage[(width * height) + (y * width) + x]);
			float b = static_cast< float >(inputImage[(2 * width * height) + (y * width) + x]);

			grayPix = ((0.3f * r) + (0.59f * g) + (0.11f * b)) + 0.5f;

			grayImage[(y * width) + x] = static_cast< unsigned char >(grayPix);
			histogram[static_cast< unsigned int >(grayPix)] += 1;
		}
	}
	// /Kernel
	kernelTime.stop();
	
	// Time GFLOP/s GB/s
	cout << fixed << setprecision(6) << kernelTime.getElapsed() << setprecision(3) << " " << (static_cast< long long unsigned int >(width) * height * 6) / 1000000000.0 / kernelTime.getElapsed() << " " << (static_cast< long long unsigned int >(width) * height * ((4 * sizeof(unsigned char)) + (1 * sizeof(unsigned int)))) / 1000000000.0 / kernelTime.getElapsed() << endl;
}