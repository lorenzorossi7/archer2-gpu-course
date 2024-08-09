/*
 * Managed memory
 *
 * An implementation of the blas level 2 routine dger(), which is
 * the operation
 *
 *   A_ij := A_ij + alpha x_i y_j
 *
 * where A is an m by n matrix, x is a vector of length m, y is
 * a vector of length n, and alpha is a constant. The data type
 * is double.
 *
 * Part 1. Replace the explicit hipMalloc()/hipMemcpy() by
 *         managed memory.
 * Part 2. Add prefetch requests for x and y before the kernel,
 *         and the matrix a after the kernel.
 *
 * Copyright EPCC, The University of Edinburgh, 2023
 */

#include <cassert>
#include <cfloat>
#include <iomanip>
#include <iostream>
#include <string>

#include "hip/hip_runtime.h"

__host__ void myErrorHandler(hipError_t ifail, std::string file, int line,
                             int fatal);

#define HIP_ASSERT(call)                                                       \
  { myErrorHandler((call), __FILE__, __LINE__, 1); }

/* Kernel parameters */

#define THREADS_PER_BLOCK_1D 256
#define THREADS_PER_BLOCK_2D 16

/* Kernel */

__global__ void myKernel(int mrow, int ncol, double alpha, double *x, double *y,
                         double *a) {

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < mrow && j < ncol) {
    a[i * ncol + j] = a[i * ncol + j] + alpha * x[i] * y[j];
  }

  return;
}

/* Main routine */

int main(int argc, char *argv[]) {

  int mrow = 1024; /* Number of rows */
  int ncol = 512;  /* Number of columns */

  double alpha = 2.0;
  double *x = NULL;
  double *y = NULL;
  double *a = NULL;

  //double *d_x = NULL;
  //double *d_y = NULL;
  //double *d_a = NULL;
  
  /* Check we have a GPU, and get device name from the hipDeviceProp
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

  /* Establish both host and device data (with some initial values for x and y) */

  //x = new double[mrow];
  //y = new double[ncol];
  //a = new double[mrow * ncol];
  //assert(x);
  //assert(y);
  //assert(a);

  HIP_ASSERT(hipMallocManaged(&x, mrow * sizeof(double)));
  HIP_ASSERT(hipMallocManaged(&y, ncol * sizeof(double)));
  HIP_ASSERT(hipMallocManaged(&a, mrow * ncol * sizeof(double)));

  for (int i = 0; i < mrow; i++) {
    x[i] = 1.0 * i;
  }
  for (int j = 0; j < ncol; j++) {
    y[j] = 1.0 * j;
  }
  for (int i = 0; i < mrow * ncol; i++) {
    a[i] = 0.0;
  }

  //HIP_ASSERT(hipMalloc(&d_x, mrow * sizeof(double)));
  //HIP_ASSERT(hipMalloc(&d_y, ncol * sizeof(double)));
  //HIP_ASSERT(hipMalloc(&d_a, mrow * ncol * sizeof(double)));

  //hipMemcpyKind kind = hipMemcpyHostToDevice;
  //HIP_ASSERT(hipMemcpy(d_x, x, mrow * sizeof(double), kind));
  //HIP_ASSERT(hipMemcpy(d_y, y, ncol * sizeof(double), kind));
  //HIP_ASSERT(hipMemset(d_a, 0, mrow * ncol * sizeof(double)));

  /* Define the execution configuration and run the kernel */
  HIP_ASSERT(hipMemPrefetchAsync(x, mrow * sizeof(double), deviceNum));
  HIP_ASSERT(hipMemPrefetchAsync(y, ncol * sizeof(double), deviceNum));

  uint nblockx = 1 + (mrow - 1) / THREADS_PER_BLOCK_2D;
  uint nblocky = 1 + (ncol - 1) / THREADS_PER_BLOCK_2D;
  dim3 blocks = {nblocky, nblockx, 1};
  dim3 threadsPerBlock = {THREADS_PER_BLOCK_2D, THREADS_PER_BLOCK_2D, 1};

  //myKernel<<<blocks, threadsPerBlock>>>(mrow, ncol, alpha, d_x, d_y, d_a);
  myKernel<<<blocks, threadsPerBlock>>>(mrow, ncol, alpha, x, y, a);

  HIP_ASSERT(hipPeekAtLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  //kind = hipMemcpyDeviceToHost;
  //HIP_ASSERT(hipMemcpy(a, d_a, mrow * ncol * sizeof(double), kind));

  int ncorrect = 0;
  std::cout << "Results:" << std::endl;
  for (int i = 0; i < mrow; i++) {
    for (int j = 0; j < ncol; j++) {
      if (fabs(a[ncol * i + j] - alpha * x[i] * y[j]) < DBL_EPSILON) {
        ncorrect += 1;
      }
    }
  }
  std::cout << "Number rows x cols " << std::setw(10) << (mrow * ncol)
            << "; correct: " << std::setw(10) << ncorrect << std::endl;

  /* Release resources */

  //HIP_ASSERT(hipFree(d_y));
  //HIP_ASSERT(hipFree(d_x));
  //HIP_ASSERT(hipFree(d_a));
  //delete a;
  //delete y;
  //delete x;
  HIP_ASSERT(hipFree(x));
  HIP_ASSERT(hipFree(y));
  HIP_ASSERT(hipFree(a));

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
