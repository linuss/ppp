
#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

float filter[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

void triangularSmooth(const int width, const int height, const int spectrum, unsigned char * inputImage, unsigned char * smoothImage) {
	NSTimer kernelTime = NSTimer("smooth", false, false);
	
	kernelTime.start();
	// Kernel
	for ( int y = 0; y < height; y++ ) {
		for ( int x = 0; x < width; x++ ) {
			for ( int z = 0; z < spectrum; z++ ) {
				unsigned int filterItem = 0;
				float filterSum = 0.0f;
				float smoothPix = 0.0f;

				for ( int fy = y - 2; fy < y + 3; fy++ ) {
					if ( fy < 0 ) {
						filterItem += 5;
						continue;
					}
					else if ( fy == height ) {
						break;
					}
					
					for ( int fx = x - 2; fx < x + 3; fx++ ) {
						if ( (fx < 0) || (fx >= width) ) {
							filterItem++;
							continue;
						}

						smoothPix += static_cast< float >(inputImage[(z * width * height) + (fy * width) + fx]) * filter[filterItem];
						filterSum += filter[filterItem];
						filterItem++;
					}
				}

				smoothPix /= filterSum;
				smoothImage[(z * width * height) + (y * width) + x] = static_cast< unsigned char >(smoothPix + 0.5f);
			}
		}
	}
	// /Kernel
	kernelTime.stop();
	
	// Time GFLOP/s GB/s
	cout << fixed << setprecision(6) << kernelTime.getElapsed() << endl;
}
