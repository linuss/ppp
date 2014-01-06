
#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

__global__ void darkenImage(const unsigned char * inputImage, 
    unsigned char * outputImage, const int width, const int height){
  int x = threadIdx.x;
  int y = threadIdx.y;

  float grayPix = 0.0f;

  float r = 
    static_cast< float >(inputImage[(y * width) + x]);
  float g = 
    static_cast< float >(inputImage[(width * height) + (y * width) + x]);
  float b = 
    static_cast< float >(inputImage[(2 * width * height) + (y * width) + x]);

  grayPix = ((0.3f * r) + (0.59f * g) + (0.11f * b));
  grayPix = (grayPix * 0.6f) + 0.5f;

  outputImage[(y * width) + x] = static_cast< unsigned char >(grayPix);
}



void darkGray(const int width, const int height, 
    const unsigned char * inputImage, unsigned char * darkGrayImage) {
	NSTimer kernelTime = NSTimer("darker", false, false);


  
  //Allocate vectors in device memory
  int color_image_size = width*height*3;
  int bw_image_size = width*height;
  unsigned char * d_input;
  cudaMalloc(&d_input, color_image_size);
  unsigned char * d_output;
  cudaMalloc(&d_output, bw_image_size);

  //Copy vector from host memory to device memory
  cudaMemcpy(d_input, inputImage, color_image_size , cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, darkGrayImage, bw_image_size , cudaMemcpyHostToDevice);




  int numBlocks = 1;
  dim3 threadsPerBlock(width,height);
	kernelTime.start();
  darkenImage<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width,
      height);
	kernelTime.stop();

  //Copy vector from device memory to host memory
  cudaMemcpy(darkGrayImage, d_output, bw_image_size, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
	
	// Time GFLOP/s GB/s
	cout << fixed << setprecision(6) << kernelTime.getElapsed() << 
    setprecision(3) << " " << (static_cast< long long unsigned int >(width) 
        * height * 7) / 1000000000.0 / kernelTime.getElapsed() << " " << 
    (static_cast< long long unsigned int >(width) * height * 
     (4 * sizeof(unsigned char))) / 1000000000.0 / kernelTime.getElapsed() 
    << endl;
}
