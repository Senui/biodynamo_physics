#!/bin/bash

# for each hardware type (CPU, OpenCL, CUDA)
for h in 0 2
do
	echo "--------------------------------------- ${h} HARDWARE"
	echo --------------------------------------- REGULAR ACCESS
	echo --------------------------------------- $n NEIGHBORS
	
        if [[ "$h" -eq 2 ]]; then
          echo "Warming GPU up..."
          echo `taskset -c 0-9,20-29 ./physics $h 128 10 20 1`
        fi
        
        for i in 16 32 64 96 128 150 180 200
	do
		echo "$i number of cells"
		echo `taskset -c 0-9,20-29 ./physics $h $i 10 38 20` >> "regular_${h}h_16n_20t.txt"
	done

	echo ""
	echo ""

	echo --------------------------------------- RANDOM ACCESS
	echo --------------------------------------- $n NEIGHBORS
	for i in 16 32 64 96 128 150 180 200
	do
		echo "$i number of cells"
		echo `taskset -c 0-9,20-29 ./physics $h $i 10 38 20 r` >> "random_${h}h_16n_20t.txt"
	done
	echo ""
done

# results in microseconds
