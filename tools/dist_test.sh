

export CUDA_VISIBLE_DEVICES=0,1,5,6

export MASTER_PORT=$((12000 + $RANDOM % 20000))
set -x

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

MKL_SERVICE_FORCE_INTEL=1 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$MASTER_PORT \
    $(dirname "$0")/test.py $CONFIG -C $CHECKPOINT --launcher pytorch ${@:4}



