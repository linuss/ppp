
#include <Timer.hpp>
#include <iostream>
#include <iomanip>
#include <cstdio>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;


const unsigned int FILTER_SIZE = 25;

__global__ void createFilterImage(unsigned char * inputImage, unsigned char* smoothImage,
    float * filter, const int width, const int height, const int spectrum){
  
  __shared__ float shared_filter[FILTER_SIZE];
  if(threadIdx.x < FILTER_SIZE){
    shared_filter[threadIdx.x] = filter[threadIdx.x];
  }
  __syncthreads();

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if(x < width && y < height){

    for ( int z = 0; z < spectrum; z++){
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

          smoothPix += static_cast< float >(inputImage[(z * width * height) + (fy * width) + fx]) * shared_filter[filterItem];
          filterSum += shared_filter[filterItem];
          filterItem++;
        }
      }
      smoothPix /= filterSum;
      smoothImage[(z * width * height) + (y * width) + x] = static_cast< unsigned char >(smoothPix + 0.5f);
    }
  }
}






float filter[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

void triangularSmooth(const int width, const int height, const int spectrum, unsigned char * inputImage, unsigned char * smoothImage) {
	NSTimer kernelTime = NSTimer("smooth", false, false);
  cudaError_t devRetVal = cudaSuccess;
  int img_size = 3 * width * height;


  //Allocate vectors in device memory
  unsigned char * d_input;
  if( (devRetVal = cudaMalloc(&d_input, img_size * 
          sizeof(unsigned char))) != cudaSuccess){
    cerr << "Impossible to allocate device memory for d_input." << endl;
    cerr << cudaGetErrorString(devRetVal) << endl;
    exit(1);
  }
  unsigned char * d_output;
  if( (devRetVal = cudaMalloc(&d_output, img_size * sizeof(unsigned char)))
      != cudaSuccess){
    cerr << "Impossible to allocate device memory for d_output." << endl;
    exit(1);
  }
  float * d_filter;
  if( (devRetVal = cudaMalloc(&d_filter, 25*sizeof(float)))
      != cudaSuccess){
    cerr << "Impossible to allocate device memory for d_filter." << endl;
    exit(1);
  }

  //Copy vector from host memory to device memory
  if( (devRetVal = cudaMemcpy(d_input, inputImage, img_size * sizeof(unsigned char) , 
          cudaMemcpyHostToDevice)) != cudaSuccess){
    cerr << "Impossible to copy inputImage to device" << endl;
    exit(1);
  }
  if( (devRetVal = cudaMemcpy(d_output, smoothImage, img_size * sizeof(unsigned char) , 
          cudaMemcpyHostToDevice)) != cudaSuccess){
    cerr << "Impossible to copy smoothImage to device" << endl;
    exit(1);
  }
  if( (devRetVal = cudaMemcpy(d_filter, filter, 25*sizeof(float) , 
          cudaMemcpyHostToDevice)) != cudaSuccess){
    cerr << "Impossible to copy filter to device" << endl;
    exit(1);
  }

  dim3 threadsPerBlock(32,32);
  dim3 numBlocks(width/threadsPerBlock.x, height/threadsPerBlock.y);

  if(width%threadsPerBlock.x != 0){
    numBlocks.x += 1;
  }
  if(height%threadsPerBlock.y != 0){
    numBlocks.y += 1;
  }

  kernelTime.start();
  createFilterImage<<<numBlocks,threadsPerBlock>>>(d_input, d_output, d_filter,
        width, height, spectrum);
  cudaDeviceSynchronize();
  kernelTime.stop();

	
  if ( ( devRetVal = cudaGetLastError()) != cudaSuccess ) {
    cerr << "Kernel has some kind of issue: " << cudaGetErrorString(devRetVal)
      << endl;
    exit(1);
  }

  //Copy vector from device memory to host memory
  if ( (devRetVal = cudaMemcpy(smoothImage, d_output, (3*height*width) * sizeof(unsigned char), 
          cudaMemcpyDeviceToHost)) != cudaSuccess){
    cerr << "Impossible to copy d_output to host " << endl;
    exit(1);
  }


  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_filter);


	// Time GFLOP/s GB/s
	cout << fixed << setprecision(6) << kernelTime.getElapsed() << 
    setprecision(3) << " " << (static_cast< long long unsigned int >(width) 
        * height * 7) / 1000000000.0 / kernelTime.getElapsed() << " " << 
    (static_cast< long long unsigned int >(width) * height * 
     (4 * sizeof(unsigned char))) / 1000000000.0 / kernelTime.getElapsed() 
    << endl;
}
