#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate data-graph-matching
PYTHON="/usr/bin/env python"

MODEL_NAME=`date +"%m_%d--%H_%M_%S"`
MODEL_DIR="./train_log/$MODEL_NAME"

mkdir -p $MODEL_DIR

TRAIN_CONFIG="MODE_FPN=True MODE_MASK=True
	DATA.VAL=('graph_val',)  DATA.TRAIN=('graph_train',) 
	TRAIN.BASE_LR=1e-3 TRAIN.EVAL_PERIOD=0 TRAIN.LR_SCHEDULE=[5000] 
	PREPROC.TRAIN_SHORT_EDGE_SIZE=[400,700] TRAIN.CHECKPOINT_PERIOD=1 DATA.NUM_WORKERS=1"

echo $TRAIN_CONFIG > $MODEL_DIR/TRAIN_CONFIG

$PYTHON train_graph.py --config $TRAIN_CONFIG --logdir ./train_log/$MODEL_NAME/log --load ../binary_files/COCO-MaskRCNN-R50FPN2x.npz