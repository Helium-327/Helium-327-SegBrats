# python inference.py --model f_cac_unet3d --data_scale full --ckpt_path \
#         "/root/workspace/Helium-327-SegBrats/results/results/2024-10-22/2024-10-22_15-28-37/checkpoints/best@e81_F_CAC_UNET3D__diceloss0.1355_dice0.8728_2024-10-22_15-28-38_15.pth"

python inference.py --model unet3d_v2 --data_scale small --data_len 4 --ckpt_path \
        "/root/workspace/Helium-327-SegBrats/结果文件夹/有效结果/2024-10-29/2024-10-29_03-13-10/checkpoints/best@e49_UNET3D_v2__diceloss0.1402_dice0.8571_2024-10-29_03-13-11_16.pth"

# 检查上一个命令的退出状态
if [ $? -eq 0 ]; then
    echo "推理完成" 
    /usr/bin/shutdown
else
    echo "debug出现错误" 
    error_time=$(date "+%Y-%m-%d %H:%M:%S")
    echo "错误时间: $error_time\n" >> debug_error.txt
    cat error.log >> debug_error.txt
fi