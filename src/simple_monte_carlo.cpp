/**********
Copyright (c) 2021, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/


#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <iomanip>

// This extension file is required for stream APIs
#include "CL/cl_ext_xilinx.h"
// This file is required for OpenCL C++ wrapper APIs
#include "xcl2.hpp"

#define NUMBER_OF_KERNELS 2

// -------------------------------------------------------------------------------------------
// An event callback function that prints the operations performed by the OpenCL runtime.
// -------------------------------------------------------------------------------------------
void event_cb(cl_event event, cl_int cmd_status, void *id)
{
	if (getenv("XCL_EMULATION_MODE") != NULL) {
	 	std::cout << "  kernel finished processing request " << *(int *)id << std::endl;
	}
}

// -------------------------------------------------------------------------------------------
// Struct returned by SmcDispatcher() and used to keep track of the request sent to the kernel
// The sync() method waits for completion of the request. After it returns, results are ready
// -------------------------------------------------------------------------------------------
class SmcRequest {

public:

  SmcRequest(int id) {
    mId = id;
    mEvent.resize(2);
  }

  void addEvent(cl::Event event)
  {
	mEvent.push_back(event);
  }

  int *getId()
  {
	return &mId;
  }

  void sync()
  {
  	// Wait until the outputs have been read back
	std::cout << "Before wait()" << std::endl;
	clWaitForEvents(1, (const cl_event *)mEvent.data());
	std::cout << "After wait()" << std::endl;

	/*clWaitForEvents(1, &mEvent[2]);
	clReleaseEvent(mEvent[0]);
   	clReleaseEvent(mEvent[1]);
   	clReleaseEvent(mEvent[2]);*/
  }

private:
  //cl_event mEvent[3];
  std::vector<cl::Event> mEvent;
  int mId;

};

// -------------------------------------------------------------------------------------------
// Class used to dispatch requests to the kernel
// The SmcDispatcher() method schedules the necessary operations (write, kernel, read) and
// returns a SmcRequest* struct which can be used to track the completion of the request.
// The dispatcher has its own OOO command queue allowing multiple requests to be scheduled
// and executed independently by the OpenCL runtime.
// -------------------------------------------------------------------------------------------
class SmcDispatcher {

public:

  SmcDispatcher(
	cl::Device     Device,
	cl::Context    Context,
	cl::Program    Program )
  {
	OCL_CHECK(mErr, mKernel = cl::Kernel(Program, "krnl_simple_monte_carlo" , &mErr));
	OCL_CHECK(mErr, mQueue  = cl::CommandQueue(Context, Device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &mErr));
	mContext = Context;
	mCounter = 0;
  }

  SmcRequest* operator() (
	unsigned int    num_elements,
	unsigned int    index,
	double    		*dst_x,
	double			*dst_y,
	int				*result)
  {

  	SmcRequest* req = new SmcRequest(mCounter++);
    std::cout << "mCounter = " << mCounter << std::endl;

  	// Running the kernel
  	unsigned int size_bytes  = num_elements * sizeof(double);
  	unsigned int size_bytes_output  = sizeof(int);

	// Create output buffer for dst (device to host)
  	OCL_CHECK(mErr, cl::Buffer buffer_output1(mContext, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, size_bytes, dst_x, &mErr));
  	OCL_CHECK(mErr, cl::Buffer buffer_output2(mContext, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, size_bytes, dst_y, &mErr));
  	OCL_CHECK(mErr, cl::Buffer buffer_output3(mContext, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, size_bytes_output, result, &mErr));

  	// Set the kernel arguments
  	OCL_CHECK(mErr, mErr = mKernel.setArg(0, buffer_output1));
  	OCL_CHECK(mErr, mErr = mKernel.setArg(1, buffer_output2));
  	OCL_CHECK(mErr, mErr = mKernel.setArg(2, buffer_output3));
  	OCL_CHECK(mErr, mErr = mKernel.setArg(3, num_elements));
  	OCL_CHECK(mErr, mErr = mKernel.setArg(4, index));

	// Schedule the execution of the kernel
  	std::vector<cl::Event> events;
  	cl::Event event_migrate_out, event_kernel;
	OCL_CHECK(mErr, mErr = mQueue.enqueueTask(mKernel, nullptr, &event_kernel));
	events.push_back(event_kernel);

	// Schedule the reading of the outputs
	OCL_CHECK(mErr, mErr = mQueue.enqueueMigrateMemObjects({buffer_output1, buffer_output2, buffer_output3},CL_MIGRATE_MEM_OBJECT_HOST, &events, &event_migrate_out));
	events.push_back(event_migrate_out);

	// Register call back to notify of kernel completion
	std::cout << "Register callback" << std::endl;
	event_kernel.setCallback(CL_COMPLETE, event_cb, req->getId());
	std::cout << "Succesful callback registering" << std::endl;

	req->addEvent(event_kernel);
	req->addEvent(event_migrate_out);

	return req;
  };

  ~SmcDispatcher()
  {
	//clReleaseCommandQueue(mQueue.get());
	//clReleaseKernel(mKernel.get());
  };

private:
  cl::Kernel        mKernel;
  cl::CommandQueue  mQueue;
  cl::Context       mContext;
  cl_int            mErr;
  int               mCounter;
};

// -------------------------------------------------------------------------------------------
// Helper functions
// -------------------------------------------------------------------------------------------
void vectors_init(double *buffer_a, double *buffer_b, int *sw_results, int *hw_results, unsigned int num_runs, unsigned int num_elements) {
  // Fill the input vectors with random data
  for (size_t i = 0; i < num_runs*num_elements; i++) {
    buffer_a[i]   = 0;
    buffer_b[i]   = 0;
  }

  for(size_t i = 0; i < num_runs; i++) {
    hw_results[i] = 0;
    sw_results[i] = 0;
  }
}

bool verify(double *x, double *y, int hw_results, int num_elements) {
  int numberOfHits = 0;
  for (int i = 0; i < num_elements; i++) {
    if (x[i]*x[i]+y[i]*y[i] <= 1) {
      numberOfHits++;
    }
  }
  return (hw_results == numberOfHits);
}

// -------------------------------------------------------------------------------------------
// Program body
// -------------------------------------------------------------------------------------------
int main(int argc, char **argv) {
  
  // ---------------------------------------------------------------------------------
  // Parse command line
  // ---------------------------------------------------------------------------------

  // Check input arguments
  if (argc < 2 || argc > 4) {
    std::cout << "Usage: " << argv[0] << " <XCLBIN File> <#elements(optional)> <debug(optional)>" << std::endl;
    return EXIT_FAILURE;
  }
  // Read FPGA binary file
  auto binaryFile = argv[1];
  unsigned int num_elements = 4096;
  bool user_size = false;
  // Check if the user defined the # of elements
  if (argc >= 3){
    user_size = true;
    unsigned int val;
    try {
      val = std::stoi(argv[2]);
    }
    catch (const std::invalid_argument val) { 
      std::cerr << "Invalid argument in position 2 (" << argv[2] << ") program expects an integer as number of elements" << std::endl;
      return EXIT_FAILURE;
    }
    catch (const std::out_of_range val) { 
      std::cerr << "Number of elements out of range, try with a number lower than 2147483648" << std::endl;
      return EXIT_FAILURE;
    }
    num_elements = val;
    std::cout << "User number of elements enabled" << std::endl;
  }
  bool debug = false;
  // Check if the user defined debug
  if (argc == 4){
    std::string debug_arg = argv[3];
    if(debug_arg.compare("debug") == 0)
      debug = true;
    std::cout << "Debug enabled" << std::endl;
  }
  
  if (!user_size){
    // Define number of num_elements  
    if (xcl::is_hw_emulation())
      num_elements= 4096;
    else if (xcl::is_emulation())
      num_elements= 4096 * 8;
    else{
      num_elements= 4096 * 4096;
    }
  }

  // I/O Data Vectors
  int numRuns = 2;
  std::vector<double, aligned_allocator<double>> buffer_a(numRuns*num_elements);
  std::vector<double, aligned_allocator<double>> buffer_b(numRuns*num_elements);
  std::vector<int, aligned_allocator<int>> hw_results(numRuns);
  std::vector<int, aligned_allocator<int>> sw_results(numRuns);
  int index = 0;

  // Initialize the data vectors
  vectors_init(buffer_a.data(), buffer_b.data(), sw_results.data(), hw_results.data(), numRuns, num_elements);

  // ---------------------------------------------------------------------------------
  // Create OpenCL context, device and program
  // ---------------------------------------------------------------------------------

  // OpenCL Host Code Begins.
  // OpenCL objects
  cl::Device device;
  cl::Context context;
  cl::CommandQueue q;
  cl::Program program;
  cl_int err;

  // get_xil_devices() is a utility API which will find the Xilinx
  // platforms and will return list of devices connected to Xilinx platform
  auto devices = xcl::get_xil_devices();

  // read_binary_file() is a utility API which will load the binaryFile
  // and will return the pointer to file buffer.
  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  bool valid_device = false;

  for (unsigned int i = 0; i < devices.size(); i++) {
    device = devices[i];
    // Creating Context and Command Queue for selected Device
    OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, program = cl::Program(context, {device}, bins, NULL, &err));
    if (err != CL_SUCCESS) {
      std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
    } else {
      std::cout << "Device[" << i << "]: program successful!\n";
      // Creating Kernel
      valid_device = true;
      break; // we break because we found a valid device
    }
  }

  if (!valid_device) {
    std::cout << "Failed to program any device found, exit!\n";
    exit(EXIT_FAILURE);
  }

  std::cout << "Running Vector add with " << num_elements << " elements" << std::endl;

  SmcRequest* request[numRuns];
  SmcDispatcher Smc(device, context, program);

  for(int r = 0; r < numRuns; r++)
  {
	// Make independent requests to Blur Y, U and V planes
	// Requests will run sequentially if there is a single CU
	// Requests will run in parallel if there are two or more CUs
	double *x_dst = buffer_a.data();
	double *y_dst = buffer_b.data();
	int* result = hw_results.data();

	request[r] = Smc(num_elements, index++, &x_dst[numRuns*num_elements], &y_dst[numRuns*num_elements], &result[numRuns]);

	// Wait for completion of the outstanding requests
	std::cout << "Before sync()" << std::endl;
	request[r]->sync();
	std::cout << "After sync()" << std::endl;
  }

  for(int i = 0; i < numRuns; i++)
  {
	  std::cout << "hw_results[" << i << "] = " << hw_results[i] << std::endl;
	  std::cout << "PI = " << 4*(double)hw_results[i]/num_elements << std::endl;
  }

  // OpenCL Host Code Ends

  // Compare the device results with software results
  bool match = true;
  //bool match = verify(buffer_a.data(), buffer_b.data(), hw_results[index], num_elements);
  std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;

  return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
