# 用于将SwinUNERT输出的nii结果转换为npz, 便于统一测试
import nibabel as nib
import numpy as np
import argparse
import os
from tqdm import tqdm
import SimpleITK as sitk
from multiprocessing import Pool
import numpy as np
from utils.display_npz import display


def nii2npz(nii_path, npz_label_dir, output_dir, phase_idx, preview_dir=None):
    case_name = nii_path.rsplit('/', 1)[-1].lstrip('CT').rstrip('.nii.gz')
    npz_name = case_name + '.npz'
    npz_label_path = os.path.join(npz_label_dir, npz_name)
    label = np.load(npz_label_path)
    ct = label["ct"]
    liver_mask = label['liver_mask']
    
    s_z= np.where(liver_mask[phase_idx] > 0)[0]
    min_z = min(s_z)
    max_z = max(s_z)
    
    out_nii = sitk.ReadImage(nii_path)
    out_array = sitk.GetArrayFromImage(out_nii)
    s, z, w, h = liver_mask.shape
    if s != 4:
        print(f'got series {s}')
    # print(f'liver:{liver_mask.shape} out:{out_array.shape}')
    
    out_array[:min_z] = 0  # 用肝mask裁剪
    out_array[max_z+1:] = 0  # 用肝mask裁剪
    pred_tumor_mask = np.zeros((s, z, w, h))
    # pred_tumor_mask[phase_idx, min_z:max_z + 1] = out_array
    pred_tumor_mask[phase_idx] = out_array
    pred_tumor_mask[pred_tumor_mask == 3] = 5
    pred_tumor_mask[pred_tumor_mask == 2] = 4
    pred_tumor_mask[pred_tumor_mask == 1] = 2
    
    output_path = os.path.join(output_dir, npz_name)
    np.savez_compressed(output_path,
                        liver_mask=liver_mask.astype(np.uint8),
                        tumor_mask=pred_tumor_mask.astype(np.uint8))
    if args.preview_dir:
        display(img=ct, mask=pred_tumor_mask, name=case_name, save_path=preview_dir)
    print(f'{case_name} done')

def main(args):
    if args.output_dir is None:
        args.output_dir = args.input_dir.rsplit('/', 1)[0] + "/outputs_npz"
    os.makedirs(args.output_dir, exist_ok=True)
    
    file_list = os.listdir(args.input_dir)
    file_list = [name for name in file_list if name.endswith('.nii.gz')]
    args_list = []
    for file_name in file_list:
        nii_path = os.path.join(args.input_dir, file_name)
        args_list.append([nii_path, args.npz_label_dir, args.output_dir, args.phase_idx, args.preview_dir])
    pool = Pool(20)
    pool.starmap(nii2npz, args_list)
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform npz datasets to nii format")
    parser.add_argument("--input_dir", default="/data4/ycy/experiment/SwinUNETR_2/outputs", type=str, help="nii results directory")
    parser.add_argument("--output_dir", help="where to save npz results")
    parser.add_argument("--npz_label_dir", default="/data4/liver_CT4_Z2_ts2.2", type=str, help="use liver mask to correcting the results")
    parser.add_argument("--preview", action="store_true", help="whether to save preview image")
    parser.add_argument("--preview_dir", default=None, help="the path to save preview image")
    parser.add_argument('--phase_idx', type=int, default=2)
    args = parser.parse_args()
    
    if args.preview and not args.preview_dir:
        args.preview_dir = args.input_dir.rsplit('/', 1)[0] + "/preview"
    if args.preview and not os.path.exists(args.preview_dir):
        os.makedirs(args.preview_dir)
    
    main(args)