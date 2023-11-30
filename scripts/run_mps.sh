#!/bin/bash

echo "+++++++++++++++++Starting the CUDA MPS-GF++++++++++++++++++"

# shellcheck disable=SC2034
# shellcheck disable=SC2126
gpu_num=$(sudo nvidia-smi|grep "Default"|wc -l)
declare gpus

# shellcheck disable=SC1073
# shellcheck disable=SC1060
for (( j = 0; j < $gpu_num; j++ )); do
    # shellcheck disable=SC2034
    gpus[$j]=$j
    sudo nvidia-smi -i $j -c EXCLUSIVE_PROCESS
    echo "Device: ${j} mps set."
done


# shellcheck disable=SC2128
export CUDA_VISIBLE_DEVICES=$gpus
# optional for cuda after 7.0, see https://developer.nvidia.com/cuda-gpus
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

nvidia-cuda-mps-control -d

echo "+++++++++++++++++MPS Started-GF++++++++++++++++++"
