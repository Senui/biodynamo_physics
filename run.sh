#!/bin/bash

## The following benchmarks runs for varying number of cells for three different
## number of neighbors

for t in 16 #1 4 16
do
	echo "--------------------------------------- ${t} THREADS"
	echo --------------------------------------- REGULAR ACCESS
	for n in 4 8 16
	do
		echo --------------------------------------- $n NEIGHBORS
		for i in 64 96 128 150 180 200 220 256 275 300 320
		do
			echo "$i number of cells"
			echo `./physics 0 $i 10 $n $t` >> "regular_${n}n_${t}t.txt"
		done
		echo ""
	done

	echo ""
	echo ""

	echo --------------------------------------- RANDOM ACCESS
	for n in 4 8 16
	do
		echo --------------------------------------- $n NEIGHBORS
		for i in 64 96 128 150 180 200 220 256 275 300 320
		do
			echo "$i number of cells"
			echo `./physics 0 $i 10 $n $t r` >> "random_${n}n_${t}t.txt"
		done
		echo ""
	done
done

## The following benchmark runs at a fixed number of cells (200^3), for
## varying number of neighbors
## 
# echo --------------------------------------- REGULAR ACCESS
# for n in {1..32} #1 2 3 4 8 16 32
# do
# 	# echo --------------------------------------- $n NEIGHBORS
# 	echo `./physics 0 200 10 $n` >> regular.txt
# 	# echo ""
# done

# echo --------------------------------------- RANDOM ACCESS
# for n in {1..32} #2 4 8 16 32 
# do
# 	# echo --------------------------------------- $n NEIGHBORS
# 	echo `./physics 0 200 10 $n r` >> random.txt
# 	# echo ""
# done
