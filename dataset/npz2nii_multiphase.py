"""
    将四期的npz文件转换成nii.gz格式的数据
"""
import numpy as np
import argparse
import os
import SimpleITK as sitk
from multiprocessing import Pool
from collections import Counter
import ants
import gc


def resample_spacing(image, type, new_spacing=(1.0, 1.0, 1.0)):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    original_direction = image.GetDirection()
    original_origin = image.GetOrigin()

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)

    resampler.SetOutputDirection(original_direction)
    resampler.SetOutputOrigin(original_origin)

    new_spacing = np.array(new_spacing, float)
    newSize = original_size / new_spacing * original_spacing
    newSize = newSize.astype(int)

    resampler.SetSize(newSize.tolist())
    if type == 'image':
        resampler.SetInterpolator(sitk.sitkBSpline)
    elif type == 'label':
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_image = resampler.Execute(image)

    return resampled_image


def npz2nii(npz_path, image_outdir, label_outdir, label_liver_outdir):
    data = np.load(npz_path)
    image = data['ct']
    label = data['tumor_mask']
    label_liver = data['liver_mask']
    spacing = data['spacing']

    if len(label) < 4 or len(label_liver) < 4:
        print(f'skip {npz_path}, for only have {len(label)} series')
        return

    spacing_counter = Counter([tuple(s) for s in spacing])
    common_spacing, _ = spacing_counter.most_common(1)[0]

    # 0:背景 1：肝正常 2：恶性肿瘤 3：胆管细胞癌 4：血管瘤 5：囊肿 6：其他 (旧数据9：其他恶性肿瘤)
    label[label == 1] = 0
    label[(label == 2) | (label == 3) | (label == 6) | (label == 9)] = 1
    label[label == 4] = 2
    label[(label == 5) | (label == 7)] = 3

    img_slice, label_slice, label_liver_slice = [], [], []
    for i in range(4):
        img3 = sitk.GetImageFromArray(image[2].astype(np.int16), isVector=False)
        label3 = sitk.GetImageFromArray(label[2].astype(np.int8), isVector=False)
        label_liver3 = sitk.GetImageFromArray(label_liver[2].astype(np.int8), isVector=False)

        img3.SetSpacing(common_spacing)
        label3.SetSpacing(common_spacing)
        label_liver3.SetSpacing(common_spacing)

        new_img3 = resample_spacing(img3, 'image', (1.5, 1.5, 5.0))
        new_label3 = resample_spacing(label3, 'label', (1.5, 1.5, 5.0))
        new_label_liver3 = resample_spacing(label_liver3, 'label', (1.5, 1.5, 5.0))

        new_img3_gt = ants.from_numpy(sitk.GetArrayFromImage(new_img3).astype(np.float32))
        # new_label3_gt = ants.from_numpy(sitk.GetArrayFromImage(new_label3))
        # new_label_liver3_gt = ants.from_numpy(sitk.GetArrayFromImage(new_label_liver3))

        if i != 2:
            new_img = sitk.GetImageFromArray(image[i], isVector=False)
            new_label = sitk.GetImageFromArray(label[i], isVector=False)
            new_label_liver = sitk.GetImageFromArray(label_liver[i], isVector=False)
            new_img.SetSpacing(common_spacing)
            new_label.SetSpacing(common_spacing)
            new_label_liver.SetSpacing(common_spacing)

            new_img = resample_spacing(new_img, 'image', (1.5, 1.5, 5.0))
            new_label = resample_spacing(new_label, 'label', (1.5, 1.5, 5.0))
            new_label_liver = resample_spacing(new_label_liver, 'label', (1.5, 1.5, 5.0))

            new_img_ = ants.from_numpy(sitk.GetArrayFromImage(new_img).astype(np.float32))
            new_label_ = ants.from_numpy(sitk.GetArrayFromImage(new_label).astype(np.float32))
            new_label_liver_ = ants.from_numpy(sitk.GetArrayFromImage(new_label_liver).astype(np.float32))

            mytx = ants.registration(fixed=new_img3_gt, moving=new_img_, type_of_transform='SyN'
                                     # , grad_step=0.2, flow_sigma=3,
                                     # total_sigma=0, aff_metric='mattes', aff_sampling=32, aff_random_sampling_rate=0.2,
                                     # syn_metric='mattes', syn_sampling=32, reg_iterations=(5, 5, 0),
                                     # aff_iterations=(1000, 1000, 1000, 10), aff_shrink_factors=(6, 4, 2, 1),
                                     # aff_smoothing_sigmas=(3, 2, 1, 0), write_composite_transform=False, random_seed=None,
                                     # verbose=False, multivariate_extras=None, restrict_transformation=None, smoothing_in_mm=False
                                     )

            warped_img = ants.apply_transforms(fixed=new_img3_gt, moving=new_img_, transformlist=mytx['fwdtransforms'],
                                               interpolator="bSpline")

            warped_label = ants.apply_transforms(fixed=new_img3_gt, moving=new_label_, transformlist=mytx['fwdtransforms'],
                                               interpolator="nearestNeighbor")

            warped_label_liver = ants.apply_transforms(fixed=new_img3_gt, moving=new_label_liver_, transformlist=mytx['fwdtransforms'],
                                               interpolator="nearestNeighbor")

            warped_img = sitk.GetImageFromArray(warped_img[:, :, :].astype(np.int16))
            warped_label = sitk.GetImageFromArray(warped_label[:, :, :].astype(np.int8))
            warped_label_liver = sitk.GetImageFromArray(warped_label_liver[:, :, :].astype(np.int8))
            warped_img.SetSpacing((1.5, 1.5, 5.0))
            warped_label.SetSpacing((1.5, 1.5, 5.0))
            warped_label_liver.SetSpacing((1.5, 1.5, 5.0))

            img_slice.append(warped_img)
            label_slice.append(warped_label)
            label_liver_slice.append(warped_label_liver)
        elif i == 2:
            img_slice.append(new_img3)
            label_slice.append(new_label3)
            label_liver_slice.append(new_label_liver3)


    image = sitk.JoinSeries(img_slice)
    label = sitk.JoinSeries(label_slice)
    label_liver = sitk.JoinSeries(label_liver_slice)

    file_name = npz_path.split('/')[-1][:-4] + ".nii.gz"
    image_savepath = os.path.join(image_outdir, 'CT' + file_name)
    label_savepath = os.path.join(label_outdir, 'SEG' + file_name)
    label_liver_savepath = os.path.join(label_liver_outdir, 'liver_SEG' + file_name)

    sitk.WriteImage(image, image_savepath)
    sitk.WriteImage(label, label_savepath)
    sitk.WriteImage(label_liver, label_liver_savepath)
    print(f'{file_name} done')


def main(args):
    image_outdir = os.path.join(args.output_dir, 'images')
    label_outdir = os.path.join(args.output_dir, 'labels')
    label_liver_outdir = os.path.join(args.output_dir, 'liver_labels')
    os.makedirs(image_outdir, exist_ok=True)
    os.makedirs(label_outdir, exist_ok=True)
    os.makedirs(label_liver_outdir, exist_ok=True)

    file_list = os.listdir(args.input_dir)
    args_list = []
    for file_name in file_list:
        npz_path = os.path.join(args.input_dir, file_name)
        args_list.append([npz_path, image_outdir, label_outdir, label_liver_outdir])
    pool = Pool(20)
    pool.starmap(npz2nii, args_list)
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform npz datasets to nii format")
    parser.add_argument("--input_dir", default="/data4/liver_CT4_Z2_ts2.2", type=str, help="input dataset directory")
    parser.add_argument("--output_dir", default="/data4/liruocheng/dataset/liver_4phase_val_reg_spacing1.5_1.5_5", type=str,
                        help="output directory")
    args = parser.parse_args()
    main(args)
