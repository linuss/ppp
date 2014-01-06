
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <Timer.hpp>
#include <cmath>
#include <iomanip>
using std::vector;
using std::make_pair;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;
using LOFAR::NSTimer;

const unsigned int clPlatformID = 0;
const unsigned int clDeviceID = 0;

const unsigned int DIM = 16000000;
const unsigned int nrThreads = 256;


int main(void) {
	unsigned int nrDevices = 0;
	float *a = new float [DIM];
	float *b = new float [DIM];
	float *c = new float [DIM];
	cl::Buffer *devA = 0;
	cl::Buffer *devB = 0;
	cl::Buffer *devC = 0;
	cl::Event clEvent;
	cl::Kernel *kernel = 0;
	vector< cl::Platform > *platforms = new vector< cl::Platform >();
	cl::Context *context = new cl::Context();
	vector< cl::Device > *devices = new vector< cl::Device >();
	vector< cl::CommandQueue > *queues = new vector< cl::CommandQueue >();
	NSTimer globalTimer("GlobalTimer", false, false);
	NSTimer kernelTimer("KernelTimer", false, false);
	NSTimer memoryTimer("MemoryTimer", false, false);
		
	// Initialize OpenCL
	try {
		cl::Platform::get(platforms);
		cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms->at(clPlatformID))(), 0};
		*context = cl::Context(CL_DEVICE_TYPE_GPU, properties);

		*devices = context->getInfo<CL_CONTEXT_DEVICES>();
		nrDevices = devices->size();
		for ( unsigned int device = 0; device < nrDevices; device++ ) {
			queues->push_back(cl::CommandQueue(*context, devices->at(device)));
		}
	} catch ( cl::Error err ) {
		cerr << "Impossible to initialize OpenCL." << endl;
		return 1;
	}

	// Prepare input and output data structures
	for ( unsigned int i = 0; i < DIM; i++ ) {
		a[i] = static_cast< float >(i);
		b[i] = static_cast< float >(i + 1);
		c[i] = 0.0f;
	}

	// Start of the computation
	globalTimer.start();

	// Allocate OpenCL memory
	try {
		devA = new cl::Buffer(*context, CL_MEM_READ_ONLY, DIM * sizeof(float), NULL, NULL);
		devB = new cl::Buffer(*context, CL_MEM_READ_ONLY, DIM * sizeof(float), NULL, NULL);
		devC = new cl::Buffer(*context, CL_MEM_READ_WRITE, DIM * sizeof(float), NULL, NULL);
	} catch ( cl::Error err ) {
		cerr << "Impossible to allocate device memory." << endl;
		return 1;
	}

	// Copy input to device
	memoryTimer.start();
	try {
		(queues->at(clDeviceID)).enqueueWriteBuffer(*devA, CL_TRUE, 0, DIM * sizeof(float), reinterpret_cast< void * >(a), NULL, &clEvent);
		clEvent.wait();
		(queues->at(clDeviceID)).enqueueWriteBuffer(*devB, CL_TRUE, 0, DIM * sizeof(float), reinterpret_cast< void * >(b), NULL, &clEvent);
		clEvent.wait();
	} catch ( cl::Error err ) {
		cerr << "Impossible to copy memory to device." << endl;
		return 1;
	}
	memoryTimer.stop();

	// Compile the kernel
	string code = "__kernel void vectorAdd(const unsigned int DIM, __global float *a, __global float *b, __global float *c) {\n"
		"unsigned int item = (get_group_id(0) * get_local_size(0)) + get_local_id(0);\n"
		"if ( item < DIM ) {\n"
		"c[item] = a[item] + b[item];\n"
		"}\n"
		"}\n";
	try {
		cl::Program *program = 0;
		cl::Program::Sources sources(1, make_pair(code.c_str(), code.length()));
		program = new cl::Program(*context, sources, NULL);
		program->build(vector< cl::Device >(1, devices->at(clDeviceID)), "-cl-mad-enable", NULL, NULL);
		kernel = new cl::Kernel(*program, "vectorAdd", NULL);
		delete program;
	} catch ( cl::Error err ) {
		cerr << "Impossible to build and create the kernel." << endl;
		return 1;
	}

	// Execute the kernel
	cl::NDRange globalSize(static_cast< unsigned int >(ceil(DIM / static_cast< float >(nrThreads))) * nrThreads);
	cl::NDRange localSize(nrThreads);
	kernel->setArg(0, DIM);
	kernel->setArg(1, *devA);
	kernel->setArg(2, *devB);
	kernel->setArg(3, *devC);

	kernelTimer.start();
	try {
		(queues->at(clDeviceID)).enqueueNDRangeKernel(*kernel, cl::NullRange, globalSize, localSize, NULL, &clEvent);
		clEvent.wait();
	} catch ( cl::Error err ) {
		cerr << "Impossible to run the kernel" << endl;
		return 1;
	}
	kernelTimer.stop();

	// Copy the output back to host
	memoryTimer.start();
	try {
		(queues->at(clDeviceID)).enqueueReadBuffer(*devC, CL_TRUE, 0, DIM * sizeof(float), reinterpret_cast< void * >(c), NULL, &clEvent);
		clEvent.wait();
	} catch ( cl::Error err ) {
		cerr << "Impossible to copy the results back to host." << endl;
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

	return 0;
}

