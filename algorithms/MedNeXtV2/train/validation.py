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

import argparse
import os
from functools import partial
from torch.utils.data import DataLoader
from utils.utils import AverageMeter, distributed_all_gather, reduce_by_weight, resample_3d, cal_dice, unSpatialPad, not_reduce_by_weight
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer7 import run_training

from utils.data_utils import get_loader, get_loader_mp

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

import nibabel as nib
from networks.MedNeXt import MedNeXt
from networks.MedNeXt_multiphase import MedNeXt_multiphase
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
from utils.utils import info_if_main, load_model, logger_info, load_pretrained_weights_resampling
from loss.deep_supervision_loss import Deep_supervision_Loss
from loguru import logger

parser = argparse.ArgumentParser(description="MedNeXt segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="/data4/liruocheng/checkpoint/MedNeXt/16", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--pretrained_dir", default="/data4/liruocheng/checkpoint/MedNeXt/16/model.pt", type=str, help="pretrained checkpoint directory")
parser.add_argument("--data_dir", default="/data4/liruocheng/dataset", type=str, help="dataset directory")
parser.add_argument("--output_directory", default='/data4/liruocheng/checkpoint/MedNeXt/16/validation', help="use tta")

parser.add_argument("--json_list", default="dataset_liver4.json", type=str, help="dataset json file")

parser.add_argument("--save_checkpoint", default=True, help="save checkpoint during training")
parser.add_argument("--max_epochs", default=500, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size(window size)")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adam", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-4, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", default=False, help="do NOT use amp for training")
parser.add_argument("--val_every", default=10, type=int, help="validation frequency")
parser.add_argument("--distributed", default=True, help="start distributed training")
parser.add_argument("--deep_supervision", default=False, help="deep supervision")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:7778", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=32, type=int, help="number of output channels")
parser.add_argument("--num_samples", default=4, type=int, help="number of output channels")
parser.add_argument("--use_normal_dataset", default=True, help="use monai Dataset class")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=5.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=32, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--use_ssl_pretrained", default=False, help="use self-supervised pretrained weights")
parser.add_argument("--upkern_weights", default=False, help="use upkern pretrained weights")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")
parser.add_argument("--num_class", default=4, help="use squared Dice")
parser.add_argument("--use_tta", default=True, help="use tta")


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logger.add(os.path.join(args.logdir, 'train_log.txt'))
    info_if_main('args:', args)
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        info_if_main(f"Found total gpus {args.ngpus_per_node}")
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    logger.add(os.path.join(args.logdir, "test_log.txt"))

    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
        torch.multiprocessing.set_sharing_strategy('file_system')
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )

    args.test_mode = False
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True

    loader = get_loader_mp(args)

    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    # MedNeXt-B
    num_block = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    scale = [2, 3, 4, 4, 4, 4, 4, 3, 2, 2]  # MedNeXt-B
    kernel_size = 3

    model = MedNeXt_multiphase(
        in_channel=args.in_channels,
        base_c=args.out_channels,
        deep_supervision=args.deep_supervision,
        k_size=kernel_size,
        num_block=num_block,
        scale=scale,
        num_class=args.num_class,
    )

    model_dict = torch.load(args.pretrained_dir)["state_dict"]
    model.load_state_dict(model_dict, strict=False)
    model.eval()
    model.cuda()

    info_if_main("Use pretrained weights")

    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
        mode="gaussian"
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger_info(f"Total parameters count {pytorch_total_params}")

    data_num, dice_sum, recall_sum, precision_sum = 0, 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader[1]):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_labels = val_labels[:, 2:3]
            original_affine = batch["label_meta_dict"]["affine"][0].numpy()
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            input_shape = val_inputs.shape[2:]
            if input_shape[0] < args.roi_x or input_shape[1] < args.roi_y or input_shape[2] < args.roi_z:
                raise Exception("image size small than trainning roi")
            if args.use_tta:
                model_inferer = partial(
                    sliding_window_inference,
                    roi_size=(args.roi_x, args.roi_y, args.roi_z),
                    sw_batch_size=4,
                    predictor=model,
                    overlap=args.infer_overlap,
                    mode="gaussian"
                )
                logits = model_inferer(val_inputs)
            else:
                val_outputs = sliding_window_inference(
                    val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
                )
            val_outputs = torch.softmax(logits, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            val_outputs = unSpatialPad(val_outputs, val_inputs.cpu().numpy()[0, 0])
            val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]
            assert val_outputs.shape == val_labels.shape

            if not os.path.exists(args.output_directory):
                os.makedirs(args.output_directory)
            nib.save(
                nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine),
                os.path.join(args.output_directory, img_name)
            )

            dsc, recall, precision = cal_dice(val_outputs > 0, val_labels > 0)
            dice_sum += dsc
            recall_sum += recall
            precision_sum += precision
            data_num += 1

            info_if_main(
                "acc",
                dsc,
                "recall",
                recall,
                "precision",
                precision)

        val_avg_acc = dice_sum / data_num
        val_avg_recall = recall_sum / data_num
        val_avg_precision = precision_sum / data_num

        info_if_main(
            "Overall avg acc",
            val_avg_acc,
            "Overall avg recall",
            val_avg_recall,
            "Overall avg precision",
            val_avg_precision
        )


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '8'
    main()
