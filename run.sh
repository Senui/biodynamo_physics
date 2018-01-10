#!/bin/bash

for i in 64 96 128 150 180 200 220 256 275 300 320
do
  echo RUNNING WITH $i CELLS PER DIM.
  ./physics 0 $i 10
  ./physics 0 $i 10 r
  echo ""
  echo ""
done

