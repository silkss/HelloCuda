CC=nvcc
CUFLAGS=-arch=compute_50 --gpu-code=sm_50

device: .\src\device_info.cu
	${CC} .\src\device_info.cu ${CUFLAGS} -o .\out\device.exe

sum: .\src\vector_sum.cu
	${CC} .\src\vector_sum.cu ${CUFLAGS} -o .\out\vector_sum.exe