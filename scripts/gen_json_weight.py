"""
    对已经转换完成的数据集，生成decathlon challenge格式的json描述文件
"""
import argparse
import os
import json
from tqdm import tqdm

basic_info = {
    "description": "liver tumor",
    "labels": {
        "0": "background",
        "1": "Cancer",
        "2": "Hemangioma",
        "3": "Cyst",
    },
    "licence": "yt",
    "modality": {
        "0": "CT"
    },
    "name": "liver_tumor",
    "numTest": 0,
    "numTraining": 0,
    "reference": "Zhejiang University",
    "release": "1.0 23/08/2023",
    "tensorImageSize": "3D",
    "test": [],
    "training": [], # need
    "validation": [], # need
}

def get_list(root_dir, data_dir, weight=False):
    results = []
    
    data_dir_abs = os.path.join(root_dir, data_dir)
    images_dir_abs = os.path.join(data_dir_abs, "images")
    labels_dir_abs = os.path.join(data_dir_abs, "labels")
    if weight:
        weights_dir_abs = os.path.join(data_dir_abs, "weights")
    
    image_names = os.listdir(images_dir_abs)
    for image_name in tqdm(image_names):
        label_name = image_name.replace('CT', 'SEG')
        label_path_abs = os.path.join(labels_dir_abs, label_name)
        if not os.path.exists(label_path_abs):
            raise Exception(f"{label_path_abs} not exist")
        if weight:
            weight_name = image_name.replace('CT', 'MAP')
            weight_path_abs = os.path.join(weights_dir_abs, weight_name)
            if not os.path.exists(weight_path_abs):
                raise Exception(f"{weight_path_abs} not exist")
        
        image_path = os.path.join(data_dir, "images", image_name)
        label_path = os.path.join(data_dir, "labels", label_name)
        if weight:
            weight_path = os.path.join(data_dir, "weights", weight_name)
        if weight:
            results.append({"image":image_path, "label":label_path, "weight":weight_path})
        else:
            results.append({"image":image_path, "label":label_path})
    
    return results

def main(args):
    data = basic_info.copy()
    data["training"] = get_list(args.root_dir, args.train_dir, weight=True)
    data["validation"] = get_list(args.root_dir, args.val_dir)
    
    with open(args.output_path, "w") as f:
        json.dump(data, f)
    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform npz datasets to nii format")
    parser.add_argument("--root_dir", default="/data1/ycy/datasets/nifty", type=str, help="root directory")
    parser.add_argument("--train_dir", default="liver_CT4_Z2_tr5.0_mp_v2", type=str, help="input dataset directory")
    parser.add_argument("--val_dir", default="liver_CT4_Z2_ts2.2_mp_v2", type=str, help="output directory")
    parser.add_argument("--output_path", default="/data1/ycy/datasets/nifty/dataset_mp_v2.json")
    args = parser.parse_args()
    main(args)