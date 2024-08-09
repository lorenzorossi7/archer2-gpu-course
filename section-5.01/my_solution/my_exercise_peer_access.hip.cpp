#include <iostream>
#include <cstring>
#include <chrono>

#include "hip/hip_runtime.h"

__host__ void myErrorHandler(hipError_t ifail, std::string file, int line,
                             int fatal);

#define HIP_ASSERT(call)                                                       \
  { myErrorHandler((call), __FILE__, __LINE__, 1); }

#define ARRAY_LENGTH 20000000

int main(int argc, char *argv[]) {
	int ndevice = -1;
	HIP_ASSERT(hipGetDeviceCount(&ndevice));

	char *array = NULL;
	array = new char[ARRAY_LENGTH];
	char *d_array = NULL;
	HIP_ASSERT(hipMalloc(&d_array, ARRAY_LENGTH * sizeof(char)));

	//Copy from host to device
	auto start = std::chrono::high_resolution_clock::now();
	hipMemcpy(d_array, array, ARRAY_LENGTH * sizeof(char), hipMemcpyHostToDevice);
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> duration = end - start;
	double timeTaken = duration.count(); // Time taken in seconds
	double bandwidth = static_cast<double>(ARRAY_LENGTH) / timeTaken; // Bandwidth in bytes per second
	double bandwidthMBps = bandwidth / (1024 * 1024); // Bandwidth in MB/s
	std::cout << "Bandwidth (MBps) when copying from host to device: " << bandwidthMBps << std::endl;

	//SEEMS NOT NEEDED: Just an untimed copy to avoid timing any initialisation overhead
	//HIP_ASSERT(hipMemcpy(array, d_array, ARRAY_LENGTH * sizeof(char), hipMemcpyDeviceToHost));
	//Copy from device to host
        start = std::chrono::high_resolution_clock::now();
        hipMemcpy(array, d_array, ARRAY_LENGTH * sizeof(char), hipMemcpyDeviceToHost);
        end = std::chrono::high_resolution_clock::now();

        duration = end - start;
        timeTaken = duration.count(); // Time taken in seconds
        bandwidth = static_cast<double>(ARRAY_LENGTH) / timeTaken; // Bandwidth in bytes per second
        bandwidthMBps = bandwidth / (1024 * 1024); // Bandwidth in MB/s
        std::cout << "Bandwidth (MBps) when copying from device to host: " << bandwidthMBps << std::endl;

	//Copy from host to device and then from device to host
        start = std::chrono::high_resolution_clock::now();
        hipMemcpy(d_array, array, ARRAY_LENGTH * sizeof(char), hipMemcpyHostToDevice);
	hipMemcpy(array, d_array, ARRAY_LENGTH * sizeof(char), hipMemcpyDeviceToHost);
        end = std::chrono::high_resolution_clock::now();

        duration = end - start;
        timeTaken = duration.count(); // Time taken in seconds
        bandwidth = static_cast<double>(ARRAY_LENGTH) / timeTaken; // Bandwidth in bytes per second
        bandwidthMBps = bandwidth / (1024 * 1024); // Bandwidth in MB/s
        std::cout << "Bandwidth (MBps) when copying from host to device and then from device to host: " << bandwidthMBps << std::endl;

	int myid1 = 1;
	if (ndevice>=2) {
		//Allocate memory for array in second device
	  	HIP_ASSERT(hipSetDevice(myid1));

        	char *d_array_id1 = NULL;
        	HIP_ASSERT(hipMalloc(&d_array_id1, ARRAY_LENGTH * sizeof(char)));
		//Return to initial device
                HIP_ASSERT(hipSetDevice(0));

		//Copy from one device to another with peer access disabled
        	start = std::chrono::high_resolution_clock::now();
        	hipMemcpy(d_array_id1, d_array, ARRAY_LENGTH * sizeof(char), hipMemcpyDeviceToDevice); //sets d_array_id1=d_array
        	end = std::chrono::high_resolution_clock::now();

	        duration = end - start;
        	timeTaken = duration.count(); // Time taken in seconds
        	bandwidth = static_cast<double>(ARRAY_LENGTH) / timeTaken; // Bandwidth in bytes per second
        	bandwidthMBps = bandwidth / (1024 * 1024); // Bandwidth in MB/s
        	std::cout << "Bandwidth (MBps) when copying from one device to another with peer access disabled: " << bandwidthMBps << std::endl;

		//Copy from one device to another with peer access enabled
		int canAccessPeer=1;
		HIP_ASSERT(hipDeviceCanAccessPeer(&canAccessPeer, 0, myid1));
		HIP_ASSERT(hipDeviceEnablePeerAccess(myid1, 0));

		start = std::chrono::high_resolution_clock::now();
                hipMemcpy(d_array_id1, d_array, ARRAY_LENGTH * sizeof(char), hipMemcpyDeviceToDevice); //sets d_array_id1=d_array
                end = std::chrono::high_resolution_clock::now();

                duration = end - start;
                timeTaken = duration.count(); // Time taken in seconds
                bandwidth = static_cast<double>(ARRAY_LENGTH) / timeTaken; // Bandwidth in bytes per second
                bandwidthMBps = bandwidth / (1024 * 1024); // Bandwidth in MB/s
                std::cout << "Bandwidth (MBps) when copying from one device to another with peer access enabled: " << bandwidthMBps << std::endl;


	} else {
                std::cout << "Device " << myid1 << " not available " << std::endl;
        }
	

	HIP_ASSERT(hipFree(d_array));
	delete(array);

	return 0;

}
	
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
