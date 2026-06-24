#!/usr/bin/env bash

set -e

P=4
MT=14
KT=14
NT=14
OUT_DIR=out
INPUTS=""
C_OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --P)
      P="$2"
      shift 2
      ;;
    --Mt)
      MT="$2"
      shift 2
      ;;
    --Kt)
      KT="$2"
      shift 2
      ;;
    --Nt)
      NT="$2"
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
    --c-out)
      C_OUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

FABRIC_X_DIM=$((P + 7))
FABRIC_Y_DIM=$((P + 2))

cslc --arch=wse3 ./layout.csl --fabric-dims=${FABRIC_X_DIM},${FABRIC_Y_DIM} --fabric-offsets=4,1 \
--params=P:${P},Mt:${MT},Kt:${KT},Nt:${NT} \
--memcpy --channels=1 -o "$OUT_DIR"

RUN_ARGS=(--name "$OUT_DIR")
if [[ -n "$INPUTS" ]]; then
  STAGED_INPUTS="inputs.npz"
  cp "$INPUTS" "$STAGED_INPUTS"
  RUN_ARGS+=(--inputs "$STAGED_INPUTS")
fi
if [[ -n "$C_OUT" ]]; then
  STAGED_C_OUT="c_cerebras.npy"
  RUN_ARGS+=(--c-out "$STAGED_C_OUT")
fi

cs_python run.py "${RUN_ARGS[@]}"

if [[ -n "$C_OUT" ]]; then
  mkdir -p "$(dirname "$C_OUT")"
  cp "$STAGED_C_OUT" "$C_OUT"
fi
