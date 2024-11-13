CONFIG_PATH=./projects/DeepRCBEV/configs/TJ4D-DeepRCBEV_det3d_2x4_12e.py
CHECKPOINT_PATH=./work_dirs/TJ4D-DeepRCBEV_det3d_2x4_12e/epoch_12.pth

python tools_det3d/test.py \
--config  $CONFIG_PATH \
--checkpoint $CHECKPOINT_PATH \
--eval mAP
