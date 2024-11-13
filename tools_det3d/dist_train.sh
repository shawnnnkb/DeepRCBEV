# tmux new -s train3d
# conda activate DeepRCBEV

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-49500} # if using multi-exp should change PORT
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CUDA_VISIBLE_DEVICES="0,1" \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    --config $CONFIG \
    --launcher pytorch ${@:3}
    
# NOTE: remind train epochs in config file
# nohup python ./tools_det3d/train.py --config ./projects/LidarPointPillars/vod-lidarpointpillars_4x1_80e.py > vod-lidarpointpillars_4x1_80e.log 2>&1 &
# nohup python ./tools_det3d/train.py --config ./configs/RCFusion/vod-RCFusion_modified_1x1_12e.py > vod-RCFusion_modified_1x1_12e.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/DeepRCBEV/configs/vod-DeepRCBEV_pretrain_2x4_12e.py 4 > vod-DeepRCBEV_pretrain_2x4_12e.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/DeepRCBEV/configs/vod-DeepRCBEV_det3d_2x4_12e.py 4 > vod-DeepRCBEV_det3d_2x4_12e.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/DeepRCBEV/configs/TJ4D-DeepRCBEV_pretrain_2x4_12e.py 4 > TJ4D-DeepRCBEV_pretrain_2x4_12e.log 2>&1 &
# nohup bash ./tools_det3d/dist_train.sh ./projects/DeepRCBEV/configs/TJ4D-DeepRCBEV_det3d_2x4_12e.py 4 > TJ4D-DeepRCBEV_det3d_2x4_12e.log 2>&1 &

# nohup bash ./tools_det3d/dist_train.sh ./projects/RCdistill/configs/vod-RCdistill_teacher_pretrain_2x4_12e.py 2 > vod-RCdistill_teacher_pretrain_2x4_12e.log 2>&1 &