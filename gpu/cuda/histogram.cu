
#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

const unsigned int nrThreads = 1024;

__global__ void createHistogram(const unsigned char * inputImage, 
    unsigned char * outputImage, unsigned int * histogram, const int img_size){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(x < (img_size)){

    float grayPix = 0.0f;

    float r = static_cast< float >(inputImage[x]);
    float g = static_cast< float >(inputImage[(img_size) + x]);
    float b = static_cast< float >(inputImage[2*(img_size) + x]);

	  grayPix = ((0.3f * r) + (0.59f * g) + (0.11f * b)) + 0.5f;

    outputImage[x] = static_cast< unsigned char >(grayPix);
    atomicAdd(&histogram[static_cast< unsigned int >(grayPix)], 1);
  }
}

void histogram1D(const int width, const int height, const unsigned char * inputImage, unsigned char * grayImage, unsigned int * histogram, unsigned char * histogramImage) {
	NSTimer kernelTime = NSTimer("histogram", false, false);
  cudaError_t devRetVal = cudaSuccess;
  int color_image_size = width*height*3;
  int bw_image_size = width*height;

  
  //Allocate vectors in device memory
  unsigned char * d_input;
  if( (devRetVal = cudaMalloc(&d_input, color_image_size * 
          sizeof(unsigned char))) != cudaSuccess){
    cerr << "Impossible to allocate device memory for d_input." << endl;
    cerr << cudaGetErrorString(devRetVal) << endl;
    exit(1);
  }
  unsigned char * d_output;
  if( (devRetVal = cudaMalloc(&d_output, bw_image_size*sizeof(unsigned char)))
      != cudaSuccess){
    cerr << "Impossible to allocate device memory for d_output." << endl;
    exit(1);
  }
  unsigned int * d_histogram;
  if( (devRetVal = cudaMalloc(&d_histogram, 256*sizeof(unsigned int)))
      != cudaSuccess){
    cerr << "Impossible to allocate device memory for d_histogram." << endl;
    exit(1);
  }


  //Copy vector from host memory to device memory
  if( (devRetVal = cudaMemcpy(d_input, inputImage, color_image_size , 
          cudaMemcpyHostToDevice)) != cudaSuccess){
    cerr << "Impossible to copy inputImage to device" << endl;
    exit(1);
  }
  if( (devRetVal = cudaMemcpy(d_output, grayImage, bw_image_size , 
          cudaMemcpyHostToDevice)) != cudaSuccess){
    cerr << "Impossible to copy grayImage to device" << endl;
    exit(1);
  }
  if( (devRetVal = cudaMemcpy(d_histogram, histogram, 256*sizeof(unsigned int) , 
          cudaMemcpyHostToDevice)) != cudaSuccess){
    cerr << "Impossible to copy histogram to device" << endl;
    exit(1);
  }

  dim3 threadsPerBlock(nrThreads);
  dim3 numBlocks(bw_image_size/nrThreads);

	
	kernelTime.start();
	// Kernel
  createHistogram<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_histogram,
      width*height);
  cudaDeviceSynchronize();
  /*
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
  */
	// /Kernel
	kernelTime.stop();
	
  if ( ( devRetVal = cudaGetLastError()) != cudaSuccess ) {
    cerr << "Kernel has some kind of issue: " << cudaGetErrorString(devRetVal)
      << endl;
    exit(1);
  }

  //Copy vector from device memory to host memory
  if ( (devRetVal = cudaMemcpy(grayImage, d_output, bw_image_size, 
          cudaMemcpyDeviceToHost)) != cudaSuccess){
    cerr << "Impossible to copy d_output to host " << endl;
    exit(1);
  }
  if ( (devRetVal = cudaMemcpy(histogram, d_histogram, 256*sizeof(unsigned int), 
          cudaMemcpyDeviceToHost)) != cudaSuccess){
    cerr << "Impossible to copy d_histogram to host " << endl;
    exit(1);
  }


  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_histogram);


	// Time GFLOP/s GB/s
	cout << fixed << setprecision(6) << kernelTime.getElapsed() << setprecision(3) << " " << (static_cast< long long unsigned int >(width) * height * 6) / 1000000000.0 / kernelTime.getElapsed() << " " << (static_cast< long long unsigned int >(width) * height * ((4 * sizeof(unsigned char)) + (1 * sizeof(unsigned int)))) / 1000000000.0 / kernelTime.getElapsed() << endl;
}
