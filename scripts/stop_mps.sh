#!/bin/bash

echo "+++++++++++++++++Stopping the CUDA MPS-GF++++++++++++++++++"

echo quit | nvidia-cuda-mps-control

# the grep string is represented Exclusive Process.
# note this logic is reasonable since we set compute mode to Exclusive
# and now we set it back for each GPU.
# shellcheck disable=SC2034
# shellcheck disable=SC2126
gpu_num=$(sudo nvidia-smi|grep "E. Process"|wc -l)

# shellcheck disable=SC1073
# shellcheck disable=SC1060
for (( j = 0; j < $gpu_num; j++ )); do
    sudo nvidia-smi -i $j -c 0
    echo "Device: ${j} mps reset."
done

echo "+++++++++++++++++MPS Stopped-GF++++++++++++++++++"