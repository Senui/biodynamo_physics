#!/bin/bash

# for each hardware type (CPU, OpenCL, CUDA)
for h in 0
do
	echo "--------------------------------------- ${h} HARDWARE"
	echo --------------------------------------- REGULAR ACCESS
	echo --------------------------------------- $n NEIGHBORS
	for i in 64 96 128 150 180 200 220 256 275 300 320
	do
		echo "$i number of cells"
		echo `./physics $h $i 20 16 1` >> "regular_${h}h_16n_1t.txt"
	done

	echo ""
	echo ""

	echo --------------------------------------- RANDOM ACCESS
	echo --------------------------------------- $n NEIGHBORS
	for i in 64 96 128 150 180 200 220 256 275 300 320
	do
		echo "$i number of cells"
		echo `./physics $h $i 20 16 1 r` >> "random_${h}h_16n_1t.txt"
	done
	echo ""
done

# results in microseconds
