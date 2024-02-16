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

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
import torch.distributed as dist
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather, reduce_by_weight, resample_3d, cal_dice, unSpatialPad, not_reduce_by_weight
from utils.utils import test_single_case
from monai.data import decollate_batch
from utils.utils import cal_metric
from utils.utils import info_if_main, logger_info
from torch.nn.utils import clip_grad_norm_, clip_grad_value_


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, deep_loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    max_norm = 3.0
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        a = time.time()
        data, target = data.to(args.gpu, non_blocking=True), target.to(args.gpu, non_blocking=True)
        if args.multi_phase:
            target = target[:, 2:3]

        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            # logits = torch.clamp(logits, min=1e-7)
            if args.deep_supervision:
                loss = deep_loss_func(logits, target)
            else:
                loss = loss_func(logits, target)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # 首先取消缩放
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Epoch {epoch}, {name} gradient: {param.grad.norm()}")
        # if np.isnan(loss.cpu().detach().numpy()):
        #     import pdb
        #     pdb.set_trace()
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
        b = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            logger_info(f'idx:{idx} rank:{args.rank} acc:{acc}')
            acc = acc.cuda(args.rank)

            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)

            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                avg_acc = np.mean(run_acc.avg)
                info_if_main(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
    return run_acc.avg


def val_epoch_v2(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    data_num, dice_sum, recall_sum, precision_sum = 0, 0, 0, 0
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            if args.multi_phase:
                target = target[:, 2:3]
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)

            val_outputs = torch.softmax(logits, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            val_outputs = unSpatialPad(val_outputs, data.cpu().numpy()[0, 0])
            val_labels = target.cpu().numpy()[0, 0, :, :, :]
            assert val_outputs.shape == val_labels.shape
            # target_shape = val_labels.shape
            # val_outputs = resample_3d(val_outputs, target_shape)

            dsc, recall, precision = cal_dice(val_outputs > 0, val_labels > 0)
            dice_sum += dsc
            recall_sum += recall
            precision_sum += precision
            data_num += 1

            if args.rank == 0:
                info_if_main(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "acc",
                    dsc,
                    "recall",
                    recall,
                    "precision",
                    precision,
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()

        if args.distributed:
            dist.barrier()
            avg_dice = reduce_by_weight(dice_sum/data_num, data_num)
            avg_recall = reduce_by_weight(recall_sum / data_num, data_num)
            avg_precision = reduce_by_weight(precision_sum / data_num, data_num)
        else:
            avg_dice = not_reduce_by_weight(dice_sum/data_num, data_num)
            avg_recall = not_reduce_by_weight(recall_sum / data_num, data_num)
            avg_precision = not_reduce_by_weight(precision_sum / data_num, data_num)
    return avg_dice, avg_recall, avg_precision


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    info_if_main("Saving checkpoint", filename)


# def run_training(
#     model,
#     train_loader,
#     val_loader,
#     optimizer,
#     loss_func,
#     deep_supervision_loss,
#     acc_func,
#     args,
#     model_inferer=None,
#     scheduler=None,
#     start_epoch=0,
#     post_label=None,
#     post_pred=None,
# ):
#     writer = None
#     if args.logdir is not None and args.rank == 0:
#         writer = SummaryWriter(log_dir=args.logdir)
#         info_if_main("Writing Tensorboard logs to ", args.logdir)
#     scaler = None
#     if args.amp:
#         scaler = GradScaler()
#     val_acc_max = 0.0
#     for epoch in range(start_epoch, args.max_epochs):
#         if args.distributed:
#             train_loader.sampler.set_epoch(epoch)
#             torch.distributed.barrier()
#         info_if_main(time.ctime(), "Epoch:", epoch)
#         epoch_time = time.time()
#         train_loss = train_epoch(
#             model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, deep_loss_func=deep_supervision_loss, args=args
#         )
#         learning_rate = optimizer.param_groups[0]['lr']
#         info_if_main(
#             "Final training  {}/{}".format(epoch, args.max_epochs - 1),
#             "loss: {:.4f}".format(train_loss),
#             "lr:{:.6f}".format(learning_rate),
#             "time {:.2f}s".format(time.time() - epoch_time),
#         )
#         if args.rank == 0 and writer is not None:
#             writer.add_scalar("train_loss", train_loss, epoch)
#             writer.add_scalar("lr", learning_rate, epoch)
#         b_new_best = False
#         if (epoch + 1) % args.val_every == 0:
#             if args.distributed:
#                 torch.distributed.barrier()
#             epoch_time = time.time()
#             val_avg_acc = val_epoch_v2(
#                 model,
#                 val_loader,
#                 epoch=epoch,
#                 acc_func=acc_func,
#                 model_inferer=model_inferer,
#                 args=args,
#                 post_label=post_label,
#                 post_pred=post_pred,
#             )
#
#             # val_avg_acc = np.mean(val_avg_acc)
#
#             if args.rank == 0:
#                 info_str = "Final validation  {}/{}".format(epoch, args.max_epochs - 1)
#                 for i in range(args.num_class-1):  # n 是类别的数量
#                     info_str += "\ndice {} all:{:4f}".format(i + 1, val_avg_acc[i])
#                     # info_str += "\nhd95 {} all:{:4f}".format(i + 1, val_avg_acc[i, 1])
#                 info_str += "\ntime {:.2f}s".format(time.time() - epoch_time)
#
#                 info_if_main(info_str)
#
#                 val_avg_acc_mean = val_avg_acc[:].mean()
#                 if writer is not None:
#                     writer.add_scalar("val dice mean", val_avg_acc_mean, epoch)
#                 if val_avg_acc_mean > val_acc_max:
#                     info_if_main("new best ({:.4f} --> {:.4f}). ".format(val_acc_max, val_avg_acc_mean))
#                     val_acc_max = val_avg_acc_mean
#                     b_new_best = True
#                     if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
#                         save_checkpoint(
#                             model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
#                         )
#             if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
#                 save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
#                 if b_new_best:
#                     info_if_main("Copying to model.pt new best model!!!!")
#                     shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))
#
#         if scheduler is not None:
#             scheduler.step()
#
#     info_if_main("Training Finished !, Best Accuracy: ", val_acc_max)
#
#     return val_acc_max

def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    deep_supervision_loss,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    best_acc=0,
    post_label=None,
    post_pred=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        info_if_main("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = best_acc
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        info_if_main(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = 0
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, deep_loss_func=deep_supervision_loss, args=args
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
            val_avg_acc, val_avg_recall, val_avg_precision = val_epoch_v2(
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
            val_avg_recall = np.mean(val_avg_recall)
            val_avg_precision = np.mean(val_avg_precision)

            if args.rank == 0:
                info_if_main(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "recall",
                    val_avg_recall,
                    "precision",
                    val_avg_precision,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                    writer.add_scalar("val_recall", val_avg_recall, epoch)
                    writer.add_scalar("val_precision", val_avg_precision, epoch)
                if val_avg_acc > val_acc_max:
                    info_if_main("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    info_if_main("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    info_if_main("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max

