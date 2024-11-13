# CONFIG_PATH=./projects/RadarPillarNet/configs/vod-radarpillarnet_modified_4x1_80e.py
# CHECKPOINT_PATH=./projects/RadarPillarNet/checkpoints/RadarPillarNet.pth
# OUTPUT_NAME=vod-RadarPillarNet
# PRED_RESULTS=./tools_det3d/view-of-delft-dataset/pred_results/$OUTPUT_NAME

# CONFIG_PATH=./projects/LidarPointPillars/configs/vod-lidarpointpillars_4x1_80e.py
# CHECKPOINT_PATH=./projects/LidarPointPillars/checkpoints/LidarPointPillars.pth
# OUTPUT_NAME=vod-LidarPointPillars
# PRED_RESULTS=./tools_det3d/view-of-delft-dataset/pred_results/$OUTPUT_NAME 

CONFIG_PATH=./projects/DeepRCBEV/configs/vod-DeepRCBEV_det3d_2x4_12e.py
CHECKPOINT_PATH=./work_dirs/vod-DeepRCBEV_det3d_2x4_12e/epoch_12.pth
OUTPUT_NAME=vod-DeepRCBEV
PRED_RESULTS=./tools_det3d/view-of-delft-dataset/pred_results/$OUTPUT_NAME 

python tools_det3d/test.py \
--format-only \
--eval-options submission_prefix=$PRED_RESULTS \
--config  $CONFIG_PATH \
--checkpoint $CHECKPOINT_PATH

python tools_det3d/view-of-delft-dataset/FINAL_EVAL.py \
--pred_results $PRED_RESULTS
