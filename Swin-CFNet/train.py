from networks.Swim_CFNet import  Swim_CFNet as Model
import torch.backends.cudnn as cudnn
from train_utils import train_one_epoch, evaluate,create_lr_scheduler
from my_dataset import My_dataset
from torch.utils.tensorboard import SummaryWriter
from networks.config import get_config
import random
import os
import numpy as np
import torch
import datetime
import time
def create_model(num_classes,config,pretrain=False):
    print(num_classes)
    model=Model(num_classes=num_classes,config=config)
    if pretrain:
        model.load_from(weights=np.load(config.pretrained_path))
    return model

def main(args):
    config=get_config()
    if not args.deterministic:
        cudnn.benchmark=True
        cudnn.deterministic=False
    else:
        cudnn.benchmark=False
        cudnn.deterministic=True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #设置随机数种子，确保结果可复现
    device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size=args.batch_size
    num_classes=args.num_classes+1
    if args.batch_size != 24 and args.batch_size%6 ==0:
        args.base_lr *=args.batch_size /24
    results_file= "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    train_dataset =My_dataset(args.data_path,
                                 set="GID15",
                                 mode="train"
                                 )
    val_dataset = My_dataset(args.data_path,
                               set="GID15",
                               mode="val"
                               )
    num_workers = 0
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn,drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=2,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn,drop_last=True)

    model = create_model(num_classes=num_classes,config=config)
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    #提取需要优化的参数，晒栓除“requires_grad”属性为True的参数

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    # 这行代码初始化了GradScaler，用于混合精度训练。
    # GradScaler是PyTorch中用于自动混合精度训练的工具，可以提高训练速度和效率。
    # 如果args.amp为True，就会创建一个GradScaler对象，否则为None。
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    #设置优化器，选择SGD作为优化器
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:#检查是否启用了混合精度训练
            scaler.load_state_dict(checkpoint["scaler"])
    #从之前保存的检查点文件中恢复训练状态。
    # 如果args.resume为True，它会加载之前训练好的模型权重、优化器状态、
    # 学习率调度器状态、当前的训练轮次，
    # 并且如果启用了混合精度训练（args.amp为True），还会加载GradScaler的状态。

    # 这里通过create_lr_scheduler函数创建一个学习率调度器，用于动态调整学习率。
    # 学习率调度器根据训练的进程来调整学习率，以帮助模型更好地收敛。
    # 在这里，optimizer是要调度的优化器，len(train_loader)
    # 表示训练数据集的批次数量，args.epochs是总的训练轮次，
    # warmup = True表示是否使用学习率热身（warmup）策略。
    best_dice = 0.
    writer = SummaryWriter(log_dir="runs/GID_RSF/logs")
    start_time = time.time()
    print(model)
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr, confmat_train = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                                       lr_scheduler=lr_scheduler, print_freq=args.print_freq,
                                                       scaler=scaler)
        train_oa, train_pa, train_iou, train_miou, train_kappa,train_f1score = confmat_train.compute()
        confmat_val, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_oa, val_pa, val_iou, val_miou, val_kappa,val_f1score= confmat_val.compute()
        val_info = str(confmat_val)
        tags = ["train_mean_loss", "lr", "train_oa", "train_kappa", "train_miou"]
        writer.add_scalar(tags[0], mean_loss, epoch)
        writer.add_scalar(tags[1], lr, epoch)
        writer.add_scalar(tags[2], train_oa, epoch)
        writer.add_scalar(tags[3], train_kappa, epoch)
        writer.add_scalar(tags[4], train_miou, epoch)
        tag1s = ["val_oa", "val_kappa", "val_miou"]
        writer.add_scalar(tag1s[0], val_oa, epoch)
        writer.add_scalar(tag1s[1], val_kappa, epoch)
        writer.add_scalar(tag1s[2], val_miou, epoch)

        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + str(confmat_train)+val_info + "\n\n")

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
                save_file = {"model": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "lr_scheduler": lr_scheduler.state_dict(),
                             "epoch": epoch,
                             "args": args}
                if args.amp:
                    save_file["scaler"] = scaler.state_dict()
                torch.save(save_file, "save_weights/GID15_My_model.pth")
            else:
                continue
        else:
            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()
            torch.save(save_file, "save_weights/Segmentation/Potsdam_{}model.pth".format(epoch))
    writer.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")
    parser.add_argument("--data-path", default=r"D:/study/Datasets/", help="DRIVE root")#数据路径
    # exclude background
    parser.add_argument("--num-classes", default=9, type=int)#分割类别
    parser.add_argument('--img_size', type=int,
                        default=512, help='input patch size of network input')#图像大小
    parser.add_argument("--device", default="cuda", help="training device")#训练设备
    parser.add_argument("-b", "--batch-size", default=8, type=int)#批量大小
    parser.add_argument("--epochs", default=300, type=int, metavar="N",
                        help="number of total epochs to train")#训练批次
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')#学习率
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')#动量
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')#权重保存
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')#权重衰减
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')#频率
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")#混合精度训练
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')#开始批次
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')#GPU
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')#是否使用确定性训练
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='segmentation network learning rate')#基础学习率
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')#随机种子
    parser.add_argument('--resume', default='./save_weights/GID15_My_model.pth', help='resume from checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)