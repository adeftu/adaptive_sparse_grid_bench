CC = gcc
CFLAGS = -O3 -funroll-loops -msse3 -Wall
NVCC = nvcc
NVCCFLAGS = -arch=sm_20 --compiler-bindir ~/gcc44

all: adaptive_sparse_grid_bench adaptive_sparse_grid_bench_cuda

adaptive_sparse_grid_bench: adaptive_sparse_grid_bench.c

adaptive_sparse_grid_bench_cuda: adaptive_sparse_grid_bench_cuda.cu
	$(NVCC) $(NVCCFLAGS) adaptive_sparse_grid_bench_cuda.cu -o adaptive_sparse_grid_bench_cuda

clean:
	rm -f adaptive_sparse_grid_bench 
	rm -f adaptive_sparse_grid_bench_cuda
	rm -f *~
