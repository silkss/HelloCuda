#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define N 512

__global__ void add(int *a, int *b, int *c){
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void randomInts(int *arr, int size) {
	for (int i = 0; i < size; i++) {
		arr[i] = rand();
	}
}

int main(void) {
	srand(time(NULL));

	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	int size = N * sizeof(int);

	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	a = (int *)malloc(size); randomInts(a, N);
	b = (int *)malloc(size); randomInts(b, N);
	c = (int *)malloc(size);

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	
	add<<<N,1>>>(d_a, d_b, d_c);

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	
	printf("calculated!\n");

	for (int i = 0; i < N; i++) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

 	free(a); free(b); free(c);	
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return 0;
}
			
