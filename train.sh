echo "即将开始集合训练"

## 2024-10-12 20:28
# python main.py --epochs 200 --data_scale full --model unet3d_bn --early_stop_patience 30 --nw 4 --bs 2 --data_split True --ts 0.8 --vs 0.1 --cosine_T_max 200 --commit 'trainging on unet3d_bn'

# echo "unet3d_bn 训练完成"

# python main.py --epochs 200 --data_scale full --model unet3d_bn_res --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit 'trainging on unet3d_bn_res'

# echo "unet3d_bn_res 训练完成"

## 2024-10-13 20:10
# python main.py --epochs 1 --data_scale debug --model unet3d_bn_res --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit 'debug on  unet3d_bn_res' 
# python main.py --epochs 1 --data_scale debug --model unet3d_bn_se --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit 'debug on  unet3d_bn_se'   
# python main.py --epochs 1 --data_scale debug --model unet3d_bn_res_se --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit 'debug on  unet3d_bn_res_se' 

# # 检查上一个命令的退出状态
# if [ $? -eq 0 ]; then
#     echo "debug完成。 可以正常开始训练" 
# else
#     echo "debug出现错误" 
#     error_time=$(date "+%Y-%m-%d %H:%M:%S")
#     echo "错误时间: $error_time\n" >> debug_error.txt
#     cat error.log >> debug_error.txt
# fi

# python main.py --epochs 200 --data_scale full --model unet3d_bn_res --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit 'fully trainging on fixbug unet3d_bn_res' 
# python main.py --epochs 200 --data_scale full --model unet3d_bn_se --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit 'fully trainging on unet3d_bn_se' 
# python main.py --epochs 200 --data_scale full --model unet3d_bn_res_se --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit 'fully trainging on unet3d_bn_res_se' 

# # 检查上一个命令的退出状态
# if [ $? -eq 0 ]; then
#     echo "😃😃😃集合训练完成"
# else
#     echo "训练出现错误" 
#     error_time=$(date "+%Y-%m-%d %H:%M:%S")
#     echo "错误时间: $error_time\n" >> debug_error.txt
#     cat error.log >> debug_error.txt
# fi

## 2024-10-16 
# python main.py --epochs 200 --data_scale full --model pspnet --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit 'fully trainging on pspnet'

# echo ""
# # 检查上一个命令的退出状态
# if [ $? -eq 0 ]; then
#     echo "😃😃😃集合训练完成"
# else
#     echo "训练出现错误" 
#     error_time=$(date "+%Y-%m-%d %H:%M:%S")
#     echo "错误时间: $error_time\n" >> debug_error.txt
#     cat error.log >> debug_error.txt
# fi

# # 2024-10-17 13:27
# python main.py --epochs 200 --data_scale full --model unet3d_bn_res --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit '将 ResCBR + CBR ---> ResCBR, fully trainging on unet3d_bn_res'

# # 检查上一个命令的退出状态
# if [ $? -eq 0 ]; then
#     echo "😃😃😃集合训练完成"
# else
#     echo "训练出现错误" 
#     error_time=$(date "+%Y-%m-%d %H:%M:%S")
#     echo "错误时间: $error_time\n" >> debug_error.txt
#     cat error.log >> debug_error.txt
# fi

# 2024-10-17 21:42

# python inference.py --model unet3d_bn_se --ckpt_path "/mnt/g/OneDrive/BraTS实验结果/有效结果/2024-10-14/2024-10-14_20-36-38/checkpoints/UNet3D_BN_SE_best_ckpt@epoch145_diceloss0.1270_dice0.8906_22.pth"
# python inference.py --model_name unet3d_bn_res_se --ckpt_path "/mnt/g/OneDrive/BraTS实验结果/有效结果/2024-10-15/2024-10-15_12-41-07/checkpoints/UNet3D_ResBN_SE_best_ckpt@epoch125_diceloss0.1278_dice0.8873_24.pth"
# python inference.py --model pspnet --ckpt_path "/mnt/g/OneDrive/BraTS实验结果/有效结果/2024-10-16/2024-10-16_22-07-11/checkpoints/PSPNET_best_ckpt@epoch81_diceloss0.2345_dice0.7681_14.pth"


# # sleep 3h
# # 设置总等待时间为3小时，即10800秒
# total_seconds=$((3 * 60 * 60))

# echo "倒计时开始，等待时间：${total_seconds} 秒"

# # 使用循环进行倒计时
# while [ $total_seconds -gt 0 ]; do
#   echo -ne "剩余时间：${total_seconds} 秒\r"
#   sleep 1 # 等待1秒
#   total_seconds=$((total_seconds - 1)) # 减少一秒
# done

# echo "倒计时结束！"

# python main.py --epochs 200 --data_scale full --model unet3d_bn_res_se --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit '将 ResCBR + CBR ---> ResCBR, fully trainging on simply unet3d_bn_res_se'
# 检查上一个命令的退出状态
# if [ $? -eq 0 ]; then
#     echo "😃😃😃模型推理完成"
# else
#     echo "训练出现错误" 
#     error_time=$(date "+%Y-%m-%d %H:%M:%S")
#     echo "错误时间: $error_time\n" >> inference_error.txt
#     cat error.log >> inference_error.txt
# fi

# 2024-10-18 16:32
# python main.py --epochs 200 --data_scale full --model unet3d_resSE --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit '将 ResCBR + CBR ---> ResCBR, fully trainging on simply unet3d_resSE'
# # 检查上一个命令的退出状态
# if [ $? -eq 0 ]; then
#     echo "😃😃😃模型推理完成"
# else
#     echo "训练出现错误" 
#     error_time=$(date "+%Y-%m-%d %H:%M:%S")
#     echo "错误时间: $error_time\n" >> inference_error.txt
#     cat error.log >> inference_error.txt
# fi

# 2024-10-19 9:22
# python main.py --epochs 200 --data_scale full --model unet3d_cbam --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit '将 ResCBR + CBR ---> ResCBR, fully trainging on simply unet3d_cbam'
# # 检查上一个命令的退出状态
# if [ $? -eq 0 ]; then
#     echo "😃😃😃模型推理完成"
# else
#     echo "训练出现错误" 
#     error_time=$(date "+%Y-%m-%d %H:%M:%S")
#     echo "错误时间: $error_time\n" >> inference_error.txt
#     cat error.log >> inference_error.txt
# fi

# 2024-10-20
# python inference.py --model unet3d_bn --ckpt_path "/root/workspace/Helium-327-SegBrats/results/2024-10-19/2024-10-19_12-53-24/checkpoints/unet3d_CBAM_best_ckpt@epoch81_diceloss0.1410_dice0.8507_25.pth"
# python main.py --epochs 200 --data_scale full --model unet3d_bn --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit "fully trainging on newly unet3d_bn"

# 2024-10-21

# python main.py --epochs 200 --data_scale full --model unet3d_bn --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit "fully trainging on newly unet3d_bn"

# # 检查上一个命令的退出状态
# if [ $? -eq 0 ]; then
#     echo "😃😃😃模型推理完成"
# else
#     echo "训练出现错误" 
#     error_time=$(date "+%Y-%m-%d %H:%M:%S")
#     echo "错误时间: $error_time\n" >> inference_error.txt
#     cat error.log >> inference_error.txt
# fi

# 2024-10-21 21:39
# python main.py --epochs 1 --data_scale debug --model f_cac_unet3d --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit 'debug on  f_cac_unet3d'
# python main.py --epochs 1 --data_scale debug --model up_cac_unet3d --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit 'debug on  up_cac_unet3d'   
# python main.py --epochs 1 --data_scale debug --model down_cac_unet3d --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit 'debug on  down_cac_unet3d' 


# python main.py --epochs 200 --data_scale full --model f_cac_unet3d --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit 'fully trainging on fixbug f_cac_unet3d' 
# python main.py --epochs 200 --data_scale full --model up_cac_unet3d --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit 'fully trainging on up_cac_unet3d' 
# python main.py --epochs 200 --data_scale full --model down_cac_unet3d --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit 'fully trainging on down_cac_unet3d' 

# # 检查上一个命令的退出状态
# if [ $? -eq 0 ]; then
#     echo "debug完成。 可以正常开始训练" 
# else
#     echo "debug出现错误" 
#     error_time=$(date "+%Y-%m-%d %H:%M:%S")
#     echo "错误时间: $error_time\n" >> debug_error.txt
#     cat error.log >> debug_error.txt
# fi

python main.py --epochs 300 --data_scale full --model f_cac_unet3d --early_stop_patience 30 --nw 8 --bs 4 --cosine_T_max 300 --commit 'fully trainging on fixbug f_cac_unet3d' 
python main.py --epochs 300 --data_scale full --model up_cac_unet3d --early_stop_patience 30 --nw 8 --bs 4 --cosine_T_max 300 --commit 'fully trainging on up_cac_unet3d' 
python main.py --epochs 300 --data_scale full --model down_cac_unet3d --early_stop_patience 30 --nw 8 --bs 4 --cosine_T_max 300 --commit 'fully trainging on down_cac_unet3d' 

# 检查上一个命令的退出状态
if [ $? -eq 0 ]; then
    echo "debug完成。 可以正常开始训练" 
else
    echo "debug出现错误" 
    error_time=$(date "+%Y-%m-%d %H:%M:%S")
    echo "错误时间: $error_time\n" >> debug_error.txt
    cat error.log >> debug_error.txt
fi

# python inference.py --model down_cac_unet3d --ckpt_path "/root/workspace/Helium-327-SegBrats/results/2024-10-22/2024-10-22_11-19-43/checkpoints/best@e17_Down_CAC_UNET3D__diceloss0.2331_dice0.7989_2024-10-22_11-19-43_10.pth"
