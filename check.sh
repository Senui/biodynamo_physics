#!/bin/bash
diff cpu.txt gpu.txt &> /dev/null

if [ $? = 0 ]; then
  echo SUCCESS
else
  echo FAILED
fi
