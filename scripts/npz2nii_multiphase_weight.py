import numpy as np
import argparse
import os
import SimpleITK as sitk
from multiprocessing import Pool
from collections import Counter
from loss.medloss import cal_weight_map
    
def npz2nii(npz_path, image_outdir, label_outdir, weight_outdir, trim=False):
    data = np.load(npz_path)
    image = data['ct']
    label = data['tumor_mask']
    spacing = data['spacing']
    liver = data['liver_mask']
    
    if len(label) < 4:
        print(f'skip {npz_path}, for only have {len(label)} series')
        return
    
    if trim:
        s_z= np.where(liver > 0)[1]
        if len(s_z) == 0:
            print(f'found empty liver mask in {npz_path}')
            return
        min_z = min(s_z)
        max_z = max(s_z)
        image = image[:, min_z:max_z+1]
        label = label[:, min_z:max_z+1]
    
    spacing_counter = Counter([tuple(s) for s in spacing])
    common_spacing, _ = spacing_counter.most_common(1)[0]
    
    # 0:背景 1：肝正常 2：恶性肿瘤 3：胆管细胞癌 4：血管瘤 5：囊肿 6：其他 (旧数据9：其他恶性肿瘤)
    label[label == 1] = 0 
    label[(label == 2) | (label == 3) | (label >= 6)] = 1
    label[label == 4] = 2
    label[(label == 5)] = 3
    
    
    img_slice, label_slice, weight_slice = [], [], []
    for i in range(4):
        weight = cal_weight_map(label[i])
        weight = (weight*255).astype(np.uint8)
        
        new_img = sitk.GetImageFromArray(image[i], isVector=False)
        new_label = sitk.GetImageFromArray(label[i], isVector=False)
        new_weight = sitk.GetImageFromArray(weight, isVector=False)
        new_img.SetSpacing(common_spacing)
        new_label.SetSpacing(common_spacing)
        new_weight.SetSpacing(common_spacing)
        img_slice.append(new_img)
        label_slice.append(new_label)
        weight_slice.append(new_weight)
    
    image = sitk.JoinSeries(img_slice)
    label = sitk.JoinSeries(label_slice)
    weight = sitk.JoinSeries(weight_slice)
    
    file_name = npz_path.split('/')[-1][:-4] + ".nii.gz"
    image_savepath = os.path.join(image_outdir, 'CT' + file_name)
    label_savepath = os.path.join(label_outdir, 'SEG' + file_name)
    weight_savepath = os.path.join(weight_outdir, 'MAP' + file_name)
    
    sitk.WriteImage(image, image_savepath)
    sitk.WriteImage(label, label_savepath)
    sitk.WriteImage(weight, weight_savepath)
    print(f'{file_name} done')

def main(args):
    image_outdir = os.path.join(args.output_dir, 'images')
    label_outdir = os.path.join(args.output_dir, 'labels')
    weight_outdir = os.path.join(args.output_dir, 'weights')
    os.makedirs(image_outdir, exist_ok=True)
    os.makedirs(label_outdir, exist_ok=True)
    os.makedirs(weight_outdir, exist_ok=True)
    
    file_list = os.listdir(args.input_dir)
    args_list = []
    for file_name in file_list:
        npz_path = os.path.join(args.input_dir, file_name)
        args_list.append([npz_path, image_outdir, label_outdir, weight_outdir, args.trim])
    pool = Pool(20)
    pool.starmap(npz2nii, args_list)
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform npz datasets to nii format")
    parser.add_argument("--input_dir", default="/data4/liver_CT4_Z2_ts2.2", type=str, help="input dataset directory")
    parser.add_argument("--output_dir", default="/data1/ycy/datasets/nifty/liver_CT4_Z2_ts2.2_mp_v2_2", type=str, help="output directory")
    parser.add_argument("--trim", action="store_true", help="Whether to cut off parts other than the liver")
    args = parser.parse_args()
    main(args)