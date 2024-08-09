/*
 * Introduction.
 *
 * Implement the simple operation x := ax for a vector x of type double
 * and a constant 'a'.
 *
 * This part introduces the kernel.
 *
 * Part 1. write a kernel of prototype
 *         __global__ void mykernel(double a, double * x)
 *         which performs the relevant operation for one block.
 * Part 2. in the main part of the program, declare and initialise
 *         variables of type dim3 to hold the number of blocks, and
 *         the number of threads per block. Use one block and
 *         THREADS_PER_BLOCK in the first instance.
 * Part 3. Generalise the kernel to treat any number of blocks,
 *         and problem sizes which are not a whole number of blocks.
 *
 * Training material originally developed by James Perry and Alan Gray
 * Copyright EPCC, The University of Edinburgh, 2010-2023
 */

#include <cassert>
#include <cfloat>
#include <iostream>
#include <iomanip>
#include <string>

#include "hip/hip_runtime.h"

/* Error checking routine and macro. */

__host__ void myErrorHandler(hipError_t ifail, const std::string file, int line,
                             int fatal);

#define HIP_ASSERT(call)                                                       \
  { myErrorHandler((call), __FILE__, __LINE__, 1); }

/* The number of integer elements in the array */
#define ARRAY_LENGTH 100

/* Suggested kernel parameters */
#define NUM_BLOCKS 1
#define THREADS_PER_BLOCK 256

__global__ void myKernel(int nlen, double a, double *x);

__global__ void daxpy(int nlen, double a, const double * x, double * y);

/* Main routine */

int main(int argc, char *argv[]) {

  size_t sz = ARRAY_LENGTH * sizeof(double);

  double a = 2.0;       /* constant a */
  double *h_x = NULL;   /* input array (host) */
  double *h_y = NULL;   /* input array (host) */
  double *h_out_x = NULL; /* output array (host) */
  double *h_out_y = NULL; /* output array (host) */
  double *d_x = NULL;   /* array (device) */
  double *d_y = NULL;   /* array (device) */

  /* Check we have a GPU, and get device name from the hipDeviceProp_t
   * structure. This is for information. */

  int ndevice = 0;
  int deviceNum = -1;
  hipDeviceProp_t prop;

  HIP_ASSERT(hipGetDeviceCount(&ndevice));

  if (ndevice == 0) {
    std::cout << "No GPU available!" << std::endl;
    std::exit(0);
  }

  HIP_ASSERT(hipGetDevice(&deviceNum));
  HIP_ASSERT(hipGetDeviceProperties(&prop, deviceNum));
  std::cout << "Device " << deviceNum << " name: " << prop.name << std::endl;
  std::cout << "Maximum number of threads per block: "
            << prop.maxThreadsPerBlock << std::endl;

  /* allocate memory on host; assign some initial values */

  h_x = new double[ARRAY_LENGTH];
  h_y = new double[ARRAY_LENGTH];
  h_out_x = new double[ARRAY_LENGTH];
  h_out_y = new double[ARRAY_LENGTH];
  assert(h_x);
  assert(h_y);
  assert(h_out_x);
  assert(h_out_y);

  for (int i = 0; i < ARRAY_LENGTH; i++) {
    h_x[i] = 1.0 * i;
    h_y[i] = 0.0;
    h_out_x[i] = 0;
    h_out_y[i] = 0;
  }

  /* allocate memory on device */

  HIP_ASSERT(hipMalloc(&d_x, sz));
  HIP_ASSERT(hipMalloc(&d_y, sz));

  /* copy input array from host to GPU */

  HIP_ASSERT(hipMemcpy(d_x, h_x, sz, hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(d_y, h_y, sz, hipMemcpyHostToDevice));

  /* ... kernel will be here  ... */

  dim3 blocks;
  dim3 threadsPerBlock;
  blocks.x = 1+(ARRAY_LENGTH-1)/THREADS_PER_BLOCK;
  threadsPerBlock = THREADS_PER_BLOCK;

  myKernel<<<blocks, threadsPerBlock>>>(ARRAY_LENGTH,a,d_x);

  HIP_ASSERT( hipPeekAtLastError() );
  HIP_ASSERT( hipDeviceSynchronize() );

  daxpy<<<blocks, threadsPerBlock>>>(ARRAY_LENGTH,a,d_x,d_y);

  HIP_ASSERT( hipPeekAtLastError() );
  HIP_ASSERT( hipDeviceSynchronize() );

  /* copy the result array back to the host output array */

  HIP_ASSERT(hipMemcpy(h_out_x, d_x, sz, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_out_y, d_y, sz, hipMemcpyDeviceToHost));

  /* We can now check the results ... */
  std::cout << "Results for h_out_x:" << std::endl;
  {
    int ncorrect = 0;
    for (int i = 0; i < ARRAY_LENGTH; i++) {
      /* The print statement can be uncommented for debugging... */
      std::cout << std::setw(9) << i << " " << std::fixed
                << std::setprecision(2) << std::setw(5) << h_out_x[i]
                << std::endl;
      if (fabs(h_out_x[i] - a * h_x[i]) < DBL_EPSILON)
        ncorrect += 1;
    }
    std::cout << "No. elements " << ARRAY_LENGTH
              << ", and correct: " << ncorrect << std::endl;
  }

  std::cout << "Results for h_out_y:" << std::endl;
  {
    int ncorrect = 0;
    for (int i = 0; i < ARRAY_LENGTH; i++) {
      /* The print statement can be uncommented for debugging... */
      std::cout << std::setw(9) << i << " " << std::fixed
                << std::setprecision(2) << std::setw(5) << h_out_y[i]
                << std::endl;
      if (fabs(h_out_y[i] - (a * h_out_x[i]+h_y[i])) < DBL_EPSILON)
        ncorrect += 1;
    }
    std::cout << "No. elements " << ARRAY_LENGTH
              << ", and correct: " << ncorrect << std::endl;
  }

  /* free device buffer */

  HIP_ASSERT(hipFree(d_x));
  HIP_ASSERT(hipFree(d_y));

  /* free host buffers */

  delete h_x;
  delete h_out_x;
  delete h_y;
  delete h_out_y;

  return 0;
}

/* It is important to check the return code from API calls, so the
 * follow function/macro allow this to be done concisely as
 *
 *   HIP_ASSERT(hipRunTimeAPIFunction(...));
 *
 * Return codes may be asynchronous, and thus misleading! */

__host__ void myErrorHandler(hipError_t ifail, const std::string file, int line,
                             int fatal) {

  if (ifail != hipSuccess) {
    std::cerr << "Line " << line << " (" << file
              << "): " << hipGetErrorName(ifail) << ": "
              << hipGetErrorString(ifail) << std::endl;
    if (fatal)
      std::exit(ifail);
  }

  return;
}

__global__ void myKernel(int nlen, double a, double *x) {

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < nlen) {
      x[i] *= 2;
    }
  }

__global__ void daxpy(int nlen, double a, const double * x, double * y){

    int i = blockIdx.x*static_cast<uint32_t>(blockDim.x) + threadIdx.x;
    if (i < nlen) {
      y[i] += a*x[i];
    }

  }
