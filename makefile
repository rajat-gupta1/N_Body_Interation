Serial:
	gcc -O3 -o serial nbody_cpu_serial.c -lm
Parallel:
	gcc -fopenmp -DPARALLEL -O3 -o parallel nbody_cpu_parallel.c -lm
Cuda:
	nvcc -arch sm_70 nbody_cuda.cu -O3 -o cuda