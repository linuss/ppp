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
const unsigned int MAX_BLOCKS = 65534;


__global__ void darkenImage(const unsigned char * inputImage,
    unsigned char * outputImage, const int width, const int height, int iteration){

  int x = ((blockIdx.x * blockDim.x) + (threadIdx.x + (iteration * MAX_BLOCKS * nrThreads))) * 3;

  if(x+2 < (3 * width*height)){
    float grayPix = 0.0f;
    float r = static_cast< float >(inputImage[x]);
    float g = static_cast< float >(inputImage[x+1]);
    float b = static_cast< float >(inputImage[x+2]);

    grayPix = __fadd_rn(__fadd_rn(__fmul_rn(0.3f, r),__fmul_rn(0.59f, g)), __fmul_rn(0.11f, b));
    grayPix = fma(grayPix,0.6f,0.5f);


    outputImage[(x/3)] = static_cast< unsigned char >(grayPix);
  }
}




void darkGray(const int width, const int height, 
    const unsigned char * inputImage, unsigned char * darkGrayImage) {
	NSTimer kernelTime = NSTimer("darker", false, false);
  cudaError_t devRetVal = cudaSuccess;
  int color_image_size = width*height*3;
  int bw_image_size = width*height;

  /*Realign the values in the input image to allow 
    coalesced memory access*/
  unsigned char * inputImageCoalesced = (unsigned char *)
    (malloc(3*height*width*(sizeof(unsigned char))));
  
  for(int i = 0; i<bw_image_size;i++){
    inputImageCoalesced[i*3] = inputImage[i];
    inputImageCoalesced[(i*3)+1] = inputImage[bw_image_size + i];
    inputImageCoalesced[(i*3)+2] = inputImage[(2*bw_image_size) + i];
  }

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


  if( (devRetVal = cudaMemcpy(d_input, inputImageCoalesced, color_image_size , 
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

  free(inputImageCoalesced);

	kernelTime.start();
	if(numBlocks > MAX_BLOCKS){
	    for(int i = 0; i<=numBlocks/MAX_BLOCKS ; i++){

	      darkenImage<<<MAX_BLOCKS, threadsPerBlock>>>(d_input, d_output, width,
		  height,i);
	      cudaDeviceSynchronize();
	    }
	}else{
	  darkenImage<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width,height,0);
	  cudaDeviceSynchronize();
	  }
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
