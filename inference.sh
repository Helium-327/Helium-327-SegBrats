# python inference.py --model f_cac_unet3d --data_scale full --ckpt_path \
#         "/root/workspace/Helium-327-SegBrats/results/results/2024-10-22/2024-10-22_15-28-37/checkpoints/best@e81_F_CAC_UNET3D__diceloss0.1355_dice0.8728_2024-10-22_15-28-38_15.pth"

python inference.py --model up_cac_unet3d --data_scale full --ckpt_path \
        "/root/workspace/Helium-327-SegBrats/results/results/2024-10-22/2024-10-22_22-26-53/checkpoints/best@e50_Up_CAC_UNET3D__diceloss0.1545_dice0.8395_2024-10-22_22-26-55_12.pth"

python inference.py --model f_cac_unet3d --data_scale full --ckpt_path \
        "/root/workspace/Helium-327-SegBrats/results/results/2024-10-23/2024-10-23_02-42-57/checkpoints/best@e122_Down_CAC_UNET3D__diceloss0.1375_dice0.8785_2024-10-23_02-42-58_28.pth"
