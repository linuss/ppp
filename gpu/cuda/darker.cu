
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

__global__ void darkenImage(const unsigned char * inputImage, 
    unsigned char * outputImage, const int width, const int height){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(x < (width * height)){

    double grayPix = 0.0f;

    double r = static_cast< double >(inputImage[x]);
    double g = static_cast< double >(inputImage[(height*width) + x]);
    double b = static_cast< double >(inputImage[2*(height*width) + x]);

    grayPix = ((0.3f * r) + (0.59f * g) + (0.11f * b));
    grayPix = (grayPix * 0.6f) + 0.5f;

    outputImage[x] = static_cast< unsigned char >(grayPix);


  }
}



void darkGray(const int width, const int height, 
    const unsigned char * inputImage, unsigned char * darkGrayImage) {
	NSTimer kernelTime = NSTimer("darker", false, false);
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


  //Copy vector from host memory to device memory
  if( (devRetVal = cudaMemcpy(d_input, inputImage, color_image_size , 
          cudaMemcpyHostToDevice)) != cudaSuccess){
    cerr << "Impossible to copy inputImage to device" << endl;
    exit(1);
  }
  if( (devRetVal = cudaMemcpy(d_output, darkGrayImage, bw_image_size , 
          cudaMemcpyHostToDevice)) != cudaSuccess){
    cerr << "Impossible to copy darkGrayImage to device" << endl;
    exit(1);
  }

  int threadsPerBlock(nrThreads);
  int numBlocks((bw_image_size/nrThreads) );

  if(bw_image_size%nrThreads != 0){
    numBlocks++;
  }


	kernelTime.start();
  darkenImage<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width,
      height);
  cudaDeviceSynchronize();
	kernelTime.stop();

  if ( ( devRetVal = cudaGetLastError()) != cudaSuccess ) {
    cerr << "Kernel has some kind of issue: " << cudaGetErrorString(devRetVal)
      << endl;
    exit(1);
  }

  //Copy vector from device memory to host memory
  if ( (devRetVal = cudaMemcpy(darkGrayImage, d_output, bw_image_size, 
          cudaMemcpyDeviceToHost)) != cudaSuccess){
    cerr << "Impossible to copy d_output to host " << endl;
    exit(1);
  }

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
