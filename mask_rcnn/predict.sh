#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJ_DIR=`realpath $SCRIPT_DIR/..`
source $PROJ_DIR/manifest


eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME
echo python binary is `which python`
PYTHON="python"


CKPT_PATH="$PROJ_DIR/ckpt/model-6000"
CONFIG=`cat $PROJ_DIR/ckpt/TRAIN_CONFIG`

while getopts i:o: flag
do
    case "${flag}" in
        i) INPUTS=${OPTARG};;
        o) OUTPUT=${OPTARG};;
    esac
done


if [[ -z "$INPUTS" || -z "$OUTPUT" ]]; then
    echo "must specify input and output"
    exit 1
fi

$PYTHON $SCRIPT_DIR/predict_graph.py --predict $INPUTS --load $CKPT_PATH --config $CONFIG --predict-output $OUTPUT