echo "即将开始集合训练"

## 2024-10-12 20:28
# python main.py --epochs 200 --data_scale full --model unet3d_bn --early_stop_patience 30 --nw 4 --bs 2 --data_split True --ts 0.8 --vs 0.1 --cosine_T_max 200 --commit 'trainging on unet3d_bn'

# echo "unet3d_bn 训练完成"

# python main.py --epochs 200 --data_scale full --model unet3d_bn_res --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit 'trainging on unet3d_bn_res'

# echo "unet3d_bn_res 训练完成"

## 2024-10-13 20:10
python main.py --epochs 1 --data_scale debug --model unet3d_bn_res --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit 'debug on  unet3d_bn_res' > /dev/null 2> error.log
python main.py --epochs 1 --data_scale debug --model unet3d_bn_se --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit 'debug on  unet3d_bn_se'   
python main.py --epochs 1 --data_scale debug --model unet3d_bn_res_se --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit 'debug on  unet3d_bn_res_se' > /dev/null 2> error.log

# 检查上一个命令的退出状态
if [ $? -eq 0 ]; then
    echo "debug完成。 可以正常开始训练" 
else
    echo "debug出现错误" 
    error_time=$(date "+%Y-%m-%d %H:%M:%S")
    echo "错误时间: $error_time\n" >> debug_error.txt
    cat error.log >> debug_error.txt
fi

python main.py --epochs 200 --data_scale full --model unet3d_bn_res --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit 'fully trainging on fixbug unet3d_bn_res' > /dev/null 2> error.log
python main.py --epochs 200 --data_scale full --model unet3d_bn_se --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit 'fully trainging on unet3d_bn_se' > /dev/null 2> error.log
python main.py --epochs 200 --data_scale full --model unet3d_bn_res_se --early_stop_patience 30 --nw 4 --bs 2 --cosine_T_max 200 --commit 'fully trainging on unet3d_bn_res_se' > /dev/null 2> error.log

# 检查上一个命令的退出状态
if [ $? -eq 0 ]; then
    echo "😃😃😃集合训练完成"
else
    echo "训练出现错误" 
    error_time=$(date "+%Y-%m-%d %H:%M:%S")
    echo "错误时间: $error_time\n" >> debug_error.txt
    cat error.log >> debug_error.txt
fi
