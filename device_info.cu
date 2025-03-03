#include <stdio.h>
#include "cuda_runtime.h"

int main(void) {
	int count = 0;
	cudaDeviceProp prop;

	cudaGetDeviceCount(&count);

	printf("Count of CUDA device(s): %d\n", count);

	for (int i = 0; i < count; i++ ) {
		cudaGetDeviceProperties(&prop, i);
		printf("Name: %s\n", prop.name);
		printf("Wrap size: %d\n", prop.warpSize);
		printf("Threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max THREAD size: %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[1]);
		printf("Max GRID size: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[1]);
	}

	return 0;
}
