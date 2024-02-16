# 生成nnUNet格式的数据集
import argparse
import os
import json
import shutil
from tqdm import tqdm


meta_info = { 
    "channel_names": {  # formerly modalities
        "0": "PV",
    }, 
    "labels": {  # THIS IS DIFFERENT NOW!
        "background": 0,
        "Cancer": 1,
        "Hem": 2,
        "Cyst": 3,
    }, 
    "numTraining": 0, 
    "file_ending": ".nii.gz",
}


def main(args):
    imagesTr = os.path.join(args.output_dir, 'imagesTr')
    labelsTr = os.path.join(args.output_dir, 'labelsTr')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    
    ori_image_dir = os.path.join(args.train_dir, 'images')
    ori_label_dir = os.path.join(args.train_dir, 'labels')
    ori_images = os.listdir(ori_image_dir)
    meta_info["numTraining"] = len(ori_images)
    json_path = os.path.join(args.output_dir, 'dataset.json')
    with open(json_path, 'w') as f:
        json.dump(meta_info, f)
    
    for ori_image in tqdm(ori_images):
        ori_image_path = os.path.join(ori_image_dir, ori_image)
        case_id = ori_image.lstrip('CT').rstrip('.nii.gz')
        target_image_path = os.path.join(imagesTr, case_id + "_0000.nii.gz")
        ori_label_path = os.path.join(ori_label_dir, 'SEG'+case_id+'.nii.gz')
        target_label_path = os.path.join(labelsTr, case_id+'.nii.gz')
        shutil.copyfile(ori_image_path, target_image_path)
        shutil.copyfile(ori_label_path, target_label_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform npz datasets to nii format")
    parser.add_argument("--train_dir", default="/home/yangcunyuan/datasets/nifty/liver_CT4_Z2_tr5.2_trim/", type=str, help="train dataset directory")
    parser.add_argument("--output_dir", default="/home/yangcunyuan/datasets/nnUNet/raw/Dataset001_CT4/", type=str, help="output directory")
    
    args = parser.parse_args()
    main(args)