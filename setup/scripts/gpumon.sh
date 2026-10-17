#!/bin/bash
# Usage: gpumon.sh <out.csv>   logs memory and utilization of every GPU every 5 seconds
while true; do nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader >> "$1"; sleep 5; done
