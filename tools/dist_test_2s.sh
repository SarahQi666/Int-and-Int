




export CUDA_VISIBLE_DEVICES=0,1,5,6

export MASTER_PORT=$((12000 + $RANDOM % 20000))
set -x


CONFIG_J=$1
CONFIG_B=$2
CHECKPOINT_J=$3
CHECKPOINT_B=$4
GPUS=$5

MKL_SERVICE_FORCE_INTEL=1 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \


python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$MASTER_PORT \
    $(dirname "$0")/test_2s.py $CONFIG_J $CONFIG_B -C_j $CHECKPOINT_J -C_b $CHECKPOINT_B --launcher pytorch ${@:6}
















