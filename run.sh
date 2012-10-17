#!/bin/bash

for i in 10,10 15,7 20,6
do 
	IFS=","; set $i
	d=$1; n=$2
	
	# hierarchization
	for h in 0 1 2 3 4 5 6
	do
		./adaptive_sparse_grid_bench_cuda $d $n 1000000 $h 5 32 &> out_${d}_${n}_${h}_5_32
	done
	
	# evaluation
	for v in 0 1 2 3 4 5
	do
		for w in 32 64 96 128 160 192 224 256
		do
			./adaptive_sparse_grid_bench_cuda $d $n 1000000 6 $v $w &> out_${d}_${n}_6_${v}_${w}
		done
	done
done
