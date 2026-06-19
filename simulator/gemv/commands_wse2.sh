#!/usr/bin/env bash

set -e

KERNEL_X_DIM=4
KERNEL_Y_DIM=3
M=6
N=8
OUT_DIR=out
INPUTS=""
Y_OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel-x-dim)
      KERNEL_X_DIM="$2"
      shift 2
      ;;
    --kernel-y-dim)
      KERNEL_Y_DIM="$2"
      shift 2
      ;;
    --M)
      M="$2"
      shift 2
      ;;
    --N)
      N="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --inputs)
      INPUTS="$2"
      shift 2
      ;;
    --y-out)
      Y_OUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

FABRIC_X_DIM=$((KERNEL_X_DIM + 7))
FABRIC_Y_DIM=$((KERNEL_Y_DIM + 2))

cslc --arch=wse2 ./layout.csl --fabric-dims=${FABRIC_X_DIM},${FABRIC_Y_DIM} \
--fabric-offsets=4,1 --params=kernel_x_dim:${KERNEL_X_DIM},kernel_y_dim:${KERNEL_Y_DIM},M:${M},N:${N} \
-o "$OUT_DIR" --memcpy --channels 1

RUN_ARGS=(--name "$OUT_DIR")
if [[ -n "$INPUTS" ]]; then
  STAGED_INPUTS="inputs.npz"
  cp "$INPUTS" "$STAGED_INPUTS"
  RUN_ARGS+=(--inputs "$STAGED_INPUTS")
fi
if [[ -n "$Y_OUT" ]]; then
  STAGED_Y_OUT="y_cerebras.npy"
  RUN_ARGS+=(--y-out "$STAGED_Y_OUT")
fi

cs_python run.py "${RUN_ARGS[@]}"

if [[ -n "$Y_OUT" ]]; then
  mkdir -p "$(dirname "$Y_OUT")"
  cp "$STAGED_Y_OUT" "$Y_OUT"
fi
