
#include <iostream>
#include <Timer.hpp>
#include <cmath>
#include <iomanip>
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;
using LOFAR::NSTimer;

const unsigned int DIM = 16000000;
const unsigned int nrThreads = 256;

__global__ void vectorAdd(const unsigned int DIM, float *a, float *b, float *c) {
	unsigned int item = (blockIdx.x * blockDim.x) + threadIdx.x;

	if ( item < DIM ) {
		c[item] = a[item] + b[item];
	}
}


int main(void) {
	cudaError_t devRetVal = cudaSuccess;
	float *a = new float [DIM];
	float *b = new float [DIM];
	float *c = new float [DIM];
	float *devA = 0;
	float *devB = 0;
	float *devC = 0;
	NSTimer globalTimer("GlobalTimer", false, false);
	NSTimer kernelTimer("KernelTimer", false, false);
	NSTimer memoryTimer("MemoryTimer", false, false);

	// Prepare input and output data structures
	for ( unsigned int i = 0; i < DIM; i++ ) {
		a[i] = static_cast< float >(i);
		b[i] = static_cast< float >(i + 1);
		c[i] = 0.0f;
	}

	// Start of the computation
	globalTimer.start();

	// Allocate CUDA memory
	if ( (devRetVal = cudaMalloc(reinterpret_cast< void ** >(&devA), DIM * sizeof(float))) != cudaSuccess ) {
		cerr << "Impossible to allocate device memory for a." << endl;
		return 1;
	}
	if ( (devRetVal = cudaMalloc(reinterpret_cast< void ** >(&devB), DIM * sizeof(float))) != cudaSuccess ) {
		cerr << "Impossible to allocate device memory for b." << endl;
		return 1;
	}
	if ( (devRetVal = cudaMalloc(reinterpret_cast< void ** >(&devC), DIM * sizeof(float))) != cudaSuccess ) {
		cerr << "Impossible to allocate device memory for c." << endl;
		return 1;
	}
	// Copy input to device
	memoryTimer.start();
	if ( (devRetVal = cudaMemcpy(devA, reinterpret_cast< void * >(a), DIM * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess ) {
		cerr << "Impossible to copy devA to device." << endl;
		return 1;
	}
	if ( (devRetVal = cudaMemcpy(devB, reinterpret_cast< void * >(b), DIM * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess ) {
		cerr << "Impossible to copy devB to device." << endl;
		return 1;
	}
	memoryTimer.stop();

	// Execute the kernel
	dim3 gridSize(static_cast< unsigned int >(ceil(DIM / static_cast< float >(nrThreads))));
	dim3 blockSize(nrThreads);

	kernelTimer.start();
	vectorAdd<<< gridSize, blockSize >>>(DIM, devA, devB, devC);
	cudaDeviceSynchronize();
	kernelTimer.stop();

	// Check if the kernel returned an error
	if ( (devRetVal = cudaGetLastError()) != cudaSuccess ) {
		cerr << "Uh, the kernel had some kind of issue: " << cudaGetErrorString(devRetVal) << endl;
		return 1;
	}

	// Copy the output back to host
	memoryTimer.start();
	if ( (devRetVal = cudaMemcpy(reinterpret_cast< void * >(c), devC, DIM * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess ) {
		cerr << "Impossible to copy devC to host." << endl;
		return 1;
	}
	memoryTimer.stop();

	// End of the computation
	globalTimer.stop();

	// Check the correctness
	for ( unsigned int i = 0; i < DIM; i++ ) {
		// Not the best floating point comparison, but this is just a CUDA example
		if ( (c[i] - (a[i] + b[i])) > 0 ) {
			cerr << "This result (" << i << ") looks wrong: " << c[i] << " != " << a[i] + b[i] << endl;
			return 1;
		}
	}

	// Print the timers
	cout << fixed << setprecision(6);
	cout << endl;
	cout << "Total (s): \t" << globalTimer.getElapsed() << endl;
	cout << "Kernel (s): \t" << kernelTimer.getElapsed() << endl;
	cout << "Memory (s): \t" << memoryTimer.getElapsed() << endl;
	cout << endl;
	cout << setprecision(3);
	cout << "GFLOP/s: \t" << (DIM / kernelTimer.getElapsed()) / 1000000000.0 << endl;
	cout << "GB/s: \t\t" << ((12 * DIM) / kernelTimer.getElapsed()) / 1000000000.0 << endl;
	cout << endl;

	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);

	return 0;
}

