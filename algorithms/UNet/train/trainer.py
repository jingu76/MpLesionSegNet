# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import time
import torch.nn.functional as F
import numpy as np
# import SimpleITK as sitk
import torch
import torch.nn.parallel
import torch.utils.data.distributed
import torch.distributed as dist
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather, reduce_by_weight, resample_3d, cal_dice, unSpatialPad

from monai.data import decollate_batch

from utils.utils import info_if_main, logger_info


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):

        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        # target = target[:, 2:3]
        data = data.permute(0, 4, 1,2,3)  #交换维度
        data=torch.reshape(data,(-1,1,128,128))

        target = target.permute(0, 4, 1,2,3)  #交换维度
        target=torch.reshape(target,(-1,1,128,128))
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)
            #
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        learning_rate = optimizer.param_groups[0]['lr']
        info_if_main(
            "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "lr:{:.6f}".format(learning_rate),
            "time {:.2f}s".format(time.time() - start_time),
        )
        start_time = time.time()
    # 代替model.zero_grad(),梯度清零
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


# 验证集的函数
def val_epoch_v2(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    data_num, dice_sum = 0, 0
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            # data:[1 1 209 209 71],target:[1 1 512 512 层数]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            # target = target[:, 2:3]
            # Initialize a dummy 3D tensor volume with shape (N,C,D,H,W)
            data = data.permute(0,1,4,2,3)  #交换维度
            # data=torch.reshape(data,(-1,1,128,128))

            target = target.permute(0,1,4,2,3)  #交换维度
            # target=torch.reshape(target,(-1,1,128,128))
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data,model)
                else:
                    print("error!!!")
                    logits = model(data)
            # logits:[1 4 209 209 71]
            # 209 209 71
            val_outputs = torch.softmax(logits, 1).cpu().numpy()
            # 获取预测结果,[209 209 71]
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            # 去掉填充,选择第0期
            val_outputs = unSpatialPad(val_outputs, data.cpu().numpy()[0, 0])
            val_labels = target.cpu().numpy()[0, 0, :, :, :]
            # [512 512 *]
            target_shape = val_labels.shape
            val_outputs = resample_3d(val_outputs, target_shape)
          
            # print(val_outputs.shape)
            # print(val_labels.shape)
            # 求dice是求的完整的图像的dice  
            # 打印出来检查一下
            
            # mat2nii(val_labels>0,epoch,idx,"label.nii.gz")
            # mat2nii(val_outputs>0,epoch,idx,"predict.nii.gz")
            
            dsc = cal_dice(val_outputs>0 , val_labels>0)
            dice_sum += dsc
            data_num += 1
            # 只有0号进程会输出dice
            if args.rank == 0:
                info_if_main(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "acc",
                    dsc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
        # 这儿导致必须采用--distributed参数，avg_dice统计了各个节点的dice的平均值
        dist.barrier()
        avg_dice = reduce_by_weight(dice_sum/data_num, data_num)
    return avg_dice


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    # hhy修改
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    info_if_main("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    model_name=f"data_dir:{args.data_dir},batch_size:{args.batch_size},lr:{args.optim_lr},roi_x:{args.roi_x},roi_y:{args.roi_y},roi_z:{args.roi_z}"
    print(model_name)
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        info_if_main("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        info_if_main(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        learning_rate = optimizer.param_groups[0]['lr']
        info_if_main(
            "Final training  {}/{}".format(epoch, args.max_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "lr:{:.6f}".format(learning_rate),
            "time {:.2f}s".format(time.time() - epoch_time),
        )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("lr", learning_rate, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch_v2(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )

            val_avg_acc = np.mean(val_avg_acc)

            if args.rank == 0:
                info_if_main(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    info_if_main("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            # model, epoch, args,filename=model_name,best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                            model, epoch, args,best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    info_if_main("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))
                    # shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, model_name))


        if scheduler is not None:
            scheduler.step()

    info_if_main("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max


# # 将[x y z]的ndarray转换成nii文件保存
# def mat2nii(mat,epoch,idx,filename,save_dir="/data4/hhy/MedNeXt/res/"):
#     directory_path = os.path.join(save_dir,str(epoch))
#     # 如果是一个新的epoch
#     if idx==0:
#         # 指定目录的路径
#         directory_path = os.path.join(save_dir,str(epoch))
#         if not os.path.exists(directory_path):
#             os.mkdir(directory_path)
#     mat = mat.transpose(2, 0, 1)  # 将 [x, y, z] 转换为 [z, x, y]
#     mat=mat.astype(np.uint8)
#     # mat=mat*64# 区分不同标签对应的像素
#     nii_path=os.path.join(directory_path,str(idx)+"_"+filename)
#     nii_file = sitk.GetImageFromArray(mat)
#     sitk.WriteImage(nii_file,nii_path) # nii_path 为保存路径