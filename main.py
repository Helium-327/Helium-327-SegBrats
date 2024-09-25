import os
import torch
import argparse
from train import train
from tabulate import tabulate

from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.amp import GradScaler

from nets.unet3ds import *
from loss_function import DiceLoss, CELoss, FocalLoss
from utils.get_commits import *
from readDatasets.BraTS import BraTS21_3d
from transforms import data_transform, Compose, RandomCrop3D, Normalize, tioRandomNoise3d, tioRandomGamma3d, tioRandomFlip3d
from utils.log_writer import *
from utils.splitDataList import DataSpliter
from metrics import EvaluationMetrics


os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
scaler = GradScaler() # 混合精度训练
MetricsGo = EvaluationMetrics() # 实例化评估指标类

def main(args):
    """------------------------------------- 记录当前实验内容 --------------------------------------------"""
    exp_commit = input("请输入本次实验的更改内容: ")

    # 创建结果保存路径
    os.makedirs(args.results_root, exist_ok=True)
    results_dir = os.path.join(args.results_root, get_current_date())
    results_dir = create_folder(results_dir) # 创建时间文件夹

    logs_dir = os.path.join(results_dir, 'logs')
    logs_path = os.path.join(logs_dir, f'{get_current_date()}.log')
    os.makedirs(logs_dir, exist_ok=True)

    write_commit_file(os.path.join(results_dir,'commits.md'), exp_commit)

    """------------------------------------- 模型实例化、初始化 --------------------------------------------"""
    start_epoch = 0
    best_val_loss = float('inf')
    
    assert args.model in ['UNet3D', 'UNet_3d_22M_32', 'UNet_3d_22M_64', 'UNet_3d_48M', 'UNet_3d_90M', 'UNet3d_bn_256', 'UNet3d_bn_512', 'UNet_3d_ln', 'UNet_3d_ln2'], "Invalid model name"
    if args.model == 'UNet3D':
        model = UNet3D(4, 4, dropout_rate=0.1)
    elif args.model == 'UNet_3d_22M_64':
        model = UNet_3d_22M_64(4, 4)
    elif args.model == 'UNet_3d_48M':
        model = UNet_3d_48M(4, 4)
    elif args.model == 'UNet_3d_90M':
        model = UNet_3d_90M(4, 4)
    elif args.model == 'UNet_3d_ln':
        model = UNet_3d_ln(4, 4)
    elif args.model == 'UNet_3d_ln2':
        model = UNet_3d_ln2(4, 4)
    elif args.model == 'UNet3d_bn_256':
        model = UNet3d_bn_256(4, 4)
    elif args.model == 'UNet3d_bn_512':
        model = UNet3d_bn_512(4, 4)
    else:
        model = UNet_3d_22M_32(4, 4)
    
    init_weights_light(model)
    model.to(DEVICE)

    """------------------------------------- 断点续传 --------------------------------------------"""
    if args.resume:
        print(f"Resuming training from checkpoint {args.resume}")
        checkpoint = torch.load(args.resume)
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint {args.resume}")
        print(f"Best val loss: {best_val_loss:.4f} ✈ epoch {start_epoch}")
    
    """------------------------------------- 获取数据列表csv --------------------------------------------"""
    train_csv = os.path.join(args.data_root, "train.csv")
    val_csv = os.path.join(args.data_root, "val.csv")
    test_csv = os.path.join(args.data_root, "test.csv")
    root = args.data_root
    path_data = os.path.join(root, "BraTS2021_Training_Data")
    assert os.path.exists(root), f"{root} not exists."
    
    """------------------------------------- 划分数据集 --------------------------------------------"""
    if args.data_split:
        dataSpliter =  DataSpliter(path_data, train_split=args.ts, val_split=args.vs, seed=RANDOM_SEED)
        train_list, test_list, val_list = dataSpliter.data_split()
        dataSpliter.save_as_csv(train_list, train_csv)
        dataSpliter.save_as_csv(test_list, val_csv)
        dataSpliter.save_as_csv(val_list, test_csv)
    else:
        delattr(args, 'ts')
        delattr(args, 'vs')

    """------------------------------------- 载入数据集 --------------------------------------------"""
    TransMethods_train = data_transform(transform=Compose([RandomCrop3D(size=args.trainCropSize),    # 随机裁剪
                                                        # tioRandonCrop3d(size=CropSize),
                                                        tioRandomFlip3d(),                 # 随机翻转
                                                        # tioRandomElasticDeformation3d(),
                                                        # tioZNormalization(),               # 归一化
                                                        tioRandomNoise3d(),
                                                        tioRandomGamma3d(),    
                                                        Normalize(mean=(0.114, 0.090, 0.170, 0.096), std=(0.199, 0.151, 0.282, 0.174)),   # 标准化
                                                        # tioRandomAffine(),          # 随机旋转
                                      ]))
    
    TransMethods_val = data_transform(transform=Compose([RandomCrop3D(size=args.valCropSize),    # 随机裁剪
                                                         Normalize(mean=(0.114, 0.090, 0.170, 0.096), std=(0.199, 0.151, 0.282, 0.174)),   # 标准化
                                                         tioRandomFlip3d(),   
                                      ]))
    
    assert args.data_scale in ['debug', 'small', 'full'], "data_scale must be 'debug', 'small' or 'full'!"
    if args.data_scale == 'small':
        # 载入部分数据集
        setattr(args, 'trainSet_len', 480)
        setattr(args, 'valSet_len', 60)
        train_dataset = BraTS21_3d(train_csv, 
                                   transform=TransMethods_train,
                                   local_train=True, 
                                   length=args.trainSet_len)
        
        val_dataset   = BraTS21_3d(val_csv, 
                                   transform=TransMethods_val, 
                                   local_train=True, 
                                   length=args.valSet_len)
    elif args.data_scale == 'debug':
        # 载入部分数据集
        setattr(args, 'trainSet_len', 80)
        setattr(args, 'valSet_len', 10)
        train_dataset = BraTS21_3d(train_csv, 
                                   transform=TransMethods_train,
                                   local_train=True, 
                                   length=args.trainSet_len)

        val_dataset   = BraTS21_3d(val_csv, 
                                   transform=TransMethods_val, 
                                   local_train=True, 
                                   length=args.valSet_len)
    else:       # 载入全部数据集
        train_dataset = BraTS21_3d(train_csv, 
                                transform=TransMethods_train)
        
        val_dataset   = BraTS21_3d(val_csv, 
                                transform=TransMethods_val)
        setattr(args, 'trainSet_len', len(train_dataset))
        setattr(args, 'valSet_len', len(val_dataset))

    assert args.nw > 0 and args.nw <= 8 , "num_workers must be in (0, 8]!"
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.bs, 
                              num_workers=args.nw,
                              shuffle=True)
    
    val_loader   = DataLoader(val_dataset, 
                              batch_size=args.bs, 
                              num_workers=args.nw,
                              shuffle=False)

    """------------------------------------- 优化器 --------------------------------------------"""
    assert args.optimizer in ['AdamW', 'SGD', 'RMSprop'], \
        f"optimizer must be 'AdamW', 'SGD' or 'RMSprop', but got {args.optimizer}."
    if args.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd) # 会出现梯度爆炸或消失
    elif args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=args.lr, alpha=0.9, eps=1e-8)
    else:
        raise ValueError("optimizer must be 'AdamW', 'SGD' or 'RMSprop'.")
    

    """------------------------------------- 调度器 --------------------------------------------"""
    assert args.scheduler in ['ReduceLROnPlateau', 'CosineAnnealingLR'], \
        f"scheduler must be 'ReduceLROnPlateau' or 'CosineAnnealingLR', but got {args.scheduler}."
    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.reduce_factor, patience=args.reduce_patience)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.cosine_T_max, eta_min=args.cosine_min_lr)
        delattr(args, 'reduce_patience')
        delattr(args, 'reduce_factor')
    
    """------------------------------------- 损失函数 --------------------------------------------"""
    assert args.loss in ['DiceLoss', 'CELoss', 'FocalLoss'], \
        f"loss must be 'DiceLoss' or 'CELoss' or 'FocalLoss', but got {args.loss}."
    if args.loss == 'CELoss':
        loss_function = CELoss(loss_type=args.loss_type)
    elif args.loss == 'FocalLoss':
        loss_function = FocalLoss(loss_type=args.loss_type)
    else:
        loss_function = DiceLoss(loss_type=args.loss_type)
    
    """--------------------------------------- 输出参数列表 --------------------------------------"""
    # 将参数转换成字典,并输出参数列表
    params = vars(args) 
    params_dict = {}
    params_dict['Parameter']=[str(p[0]) for p in list(params.items())]
    params_dict['Value']=[str(p[1]) for p in list(params.items())]
    params_header = ["Parameter", "Value"]

    # 标准输出
    print(tabulate(params_dict, headers=params_header, tablefmt="grid"))
    # 重定向输出
    custom_logger('='*40 + '\n' + "训练参数" +'\n' + '='*40 +'\n', logs_path, log_time=True)
    custom_logger(tabulate(params_dict, headers=params_header, tablefmt="grid"), logs_path)
    
    train(model, 
          Metrics=MetricsGo, 
          train_loader=train_loader,
          val_loader=val_loader, 
          scaler=scaler, 
          optimizer=optimizer,
          scheduler=scheduler,
          loss_function=loss_function,
          num_epochs=args.epochs, 
          device=DEVICE, 
          results_dir=results_dir,
          logs_path=logs_path,
          start_epoch=start_epoch,
          best_val_loss=best_val_loss,
          tb=args.tb,
          interval=args.interval,
          save_max=args.save_max,
          early_stopping_patience=args.early_stop_patience)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train args")

    parser.add_argument("--data_root" , type=str, default="./brats21_local", help="data root")
    parser.add_argument("--results_root", type=str, default="./results", help="result path")
    parser.add_argument("--resume", type=str, default=None, help="resume training from checkpoint")
    
    parser.add_argument("--model", type=str, default="UNet3D", help="models: ['UNet3D', 'UNet_3d_22M_32', 'UNet_3d_22M_64', 'UNet_3d_48M', 'UNet_3d_90M', 'UNet3d_bn_256', 'UNet3d_bn_512', 'UNet_3d_ln', 'UNet_3d_ln2']")
    parser.add_argument("--epochs", type=int, default=60, help="num_epochs")
    parser.add_argument("--nw", type=int, default=8, help="num_workers")
    parser.add_argument("--bs", type=int, default=2, help="batch_size")
    parser.add_argument("--early_stop_patience", type=int, default=20, help="early stop patience")
    
    parser.add_argument("--input_channels", type=int, default=4, help="input channels")
    parser.add_argument("--output_channels", type=int, default=4, help="output channels")
    parser.add_argument("--trainCropSize", type=lambda x: tuple(map(int, x.split(','))), default=(128, 128, 128), help="crop size")
    parser.add_argument("--valCropSize", type=lambda x: tuple(map(int, x.split(','))), default=(128, 128, 128), help="crop size")
    
    parser.add_argument("--loss", type=str, default="FocalLoss", help="loss function: ['DiceLoss', 'CELoss', 'FocalLoss']")
    parser.add_argument("--loss_type", type=str, default="custom", help="loss type to grad")
    parser.add_argument("--save_max", type=int, default=5, help="ckpt max save number")

    parser.add_argument("--optimizer", type=str, default="AdamW", help="optimizers: ['AdamW', 'SGD', 'RMSprop']")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")

    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR", help="schedulers:['ReduceLROnPlateau', 'CosineAnnealingLR']")
    parser.add_argument("--cosine_min_lr", type=float, default=1e-4, help="CosineAnnealingLR min lr")
    parser.add_argument("--cosine_T_max", type=float, default=100, help="CosineAnnealingLR T max")
    # parser.add_argument("--cosine_last_epoch", type=int, default=30, help="CosineAnnealingLR last epoch")

    parser.add_argument("--reduce_patience", type=int, default=3, help="ReduceLROnPlateau scheduler patience")
    parser.add_argument("--reduce_factor", type=float, default=0.9, help="ReduceLROnPlateau scheduler factor")
    
    parser.add_argument("--data_scale", type=str, default="debug", help="loading data scale")
    parser.add_argument("--trainSet_len", type=int, default=100, help="train length")
    parser.add_argument("--valSet_len", type=int, default=12, help="val length")
    parser.add_argument("--interval", type=int, default=1, help="checkpoint interval")
    
    parser.add_argument("--tb", type=bool, default=False, help="TensorBoard True or False")
    parser.add_argument("--data_split", type=bool, default=False, help="data split True or False")
    parser.add_argument("--ts", type=float, default=0.8, help="train_split_rata")
    parser.add_argument("--vs", type=float, default=0.1, help="val_split_rate")
    # parser.add_argument("--stop_patience", type=int, default=10, help="early stopping")
    

    args = parser.parse_args()
    main(args=args)