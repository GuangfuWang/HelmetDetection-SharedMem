#!/bin/bash

echo "====================transform onnx model to tensorrt model==========================="

if [ $# -eq 1 ]
then echo "Usage: /path/to/the/onnx2tensorrt.sh <model.onnx> [optional][model.engine]"
exit
fi

# shellcheck disable=SC2034
onnx_model=$1

declare out_model
# shellcheck disable=SC1073
# shellcheck disable=SC1035
# shellcheck disable=SC1009
if [ $# -eq 3 ]
then
  out_model=$2
else
  out_model="model.engine"
fi

/usr/src/tensorrt/bin/trtexec --onnx="$onnx_model" --saveEngine="$out_model"

echo "=====================onnx model to tensorrt model success============================"