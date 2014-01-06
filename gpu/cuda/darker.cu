
#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;


void darkGray(const int width, const int height, const unsigned char * inputImage, unsigned char * darkGrayImage) {
	NSTimer kernelTime = NSTimer("darker", false, false);
	
	kernelTime.start();
	// Kernel
	for ( int y = 0; y < height; y++ ) {
		for ( int x = 0; x < width; x++ ) {
			float grayPix = 0.0f;
			float r = static_cast< float >(inputImage[(y * width) + x]);
			float g = static_cast< float >(inputImage[(width * height) + (y * width) + x]);
			float b = static_cast< float >(inputImage[(2 * width * height) + (y * width) + x]);

			grayPix = ((0.3f * r) + (0.59f * g) + (0.11f * b));
			grayPix = (grayPix * 0.6f) + 0.5f;

			darkGrayImage[(y * width) + x] = static_cast< unsigned char >(grayPix);
		}
	}
	// /Kernel
	kernelTime.stop();
	
	// Time GFLOP/s GB/s
	cout << fixed << setprecision(6) << kernelTime.getElapsed() << setprecision(3) << " " << (static_cast< long long unsigned int >(width) * height * 7) / 1000000000.0 / kernelTime.getElapsed() << " " << (static_cast< long long unsigned int >(width) * height * (4 * sizeof(unsigned char))) / 1000000000.0 / kernelTime.getElapsed() << endl;
}