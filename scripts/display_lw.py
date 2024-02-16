"""
@About : 可视化 ct 与 mask 矩阵
"""
import os
from multiprocessing.dummy import Pool
import cv2
import numpy as np
import argparse
from matplotlib.colors import ListedColormap


COLOR_TABLE = [(0, 0, 0), (255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]     # set color of different tumor
COLOR_TABLE2 = [(0, 0, 0), (255, 255, 255)]
CLS_LIS = [1, 2, 3, 4, 5, 6]         # set index of tumor to display


def get_namelist(path, condition=lambda x: 'ct_' not in x and '.zip' not in x and '.npz' in x):
    """
    按条件筛选文件
    :param path: 待筛选的文件夹
    :param condition:  过滤条件
    :return:  筛选后的文件名列表（相对路径）
    """
    file_name_list = []
    if os.path.isdir(path):
        file_name_list = os.listdir(path)
        file_name_list = [n for n in file_name_list if condition(n)]
    elif os.path.isfile(path):
        path = os.path.basename(path)
        assert 'ct_' not in path
        file_name_list.append(path)
    return file_name_list


def display(img, mask, save_path, name):
    assert len(img) == len(mask)
    result = []

    img = np.clip(img, -55, 155)
    img = (img + 55.) / (155. + 55.)

    if np.max(mask) > 6:
        print(name, "error")

    for img_single, mask_single in zip(img, mask):
        single_series = []
        for i in range(len(img_single)):
            img_slice, mask_slice = img_single[i], mask_single[i]
            img_slice = img_slice * 255
            img_slice = cv2.cvtColor(img_slice.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
            cmap = ListedColormap(COLOR_TABLE)
            color_mask = cmap(mask_slice)

            overlay_image = 0.6 * color_mask[:, :, :3] + 0.4 * img_slice.astype(float)
            overlay_image = np.clip(overlay_image, 0, 255).astype(np.uint8)

            single_series.append(overlay_image)
        result.append(cv2.vconcat(single_series))
        
    for img_single, mask_single in zip(img, mask):
        single_series = []
        for i in range(len(img_single)):
            mask_slice = mask_single[i]
            
            cmap = ListedColormap(COLOR_TABLE2)
            color_mask = cmap((mask_slice>0))
            color_mask = color_mask[:, :, :3].astype(np.uint8)
            
            single_series.append(color_mask)
        result.append(cv2.vconcat(single_series))
    
    if args.show_layer:
        layers = []
        for i in range(img.shape[1]):
            layer_i = np.zeros((512, 512, 3), dtype=np.uint8)
            text = f"{i}"
            org = (200, 200)  # 文本的起始位置
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 3
            color = (255, 255, 255)  # white
            thickness = 4
            layer_i = cv2.putText(layer_i, text, org, font, fontScale, color, thickness)
            layers.append(layer_i)
        result.append(cv2.vconcat(layers))
           
    result = cv2.hconcat(result)
    cv2.imwrite(os.path.join(save_path, name + ".png"), result, [cv2.IMWRITE_PNG_COMPRESSION, 10])
    print(f'finished save png {name}')

def process(name):
    # shape [s, n, h, w]
    # img, mask = get_img(name)
    img = np.load(os.path.join(args.ct_path, name), allow_pickle=True).get("ct")
    tumor_mask = np.load(os.path.join(args.draw_path, name), allow_pickle=True).get("tumor_mask")
    display(img, tumor_mask, args.save_path, name)


# display a dir
def draw(file_path):
    name_list = get_namelist(file_path)
    inp = [(i,) for i in name_list]
    pool = Pool(5)
    pool.starmap(process, inp)
    pool.close()
    pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="display")
    parser.add_argument('--draw_path', type=str, default='/home/yangcunyuan/datasets/sample_npz/Sample/')
    parser.add_argument('--save_path', type=str, default="/home/yangcunyuan/datasets/sample_png/")
    parser.add_argument('--ct_path', type=str, default="/home/yangcunyuan/datasets/sample_npz/Sample/")
    parser.add_argument('--show_layer', action='store_true')
    
    args = parser.parse_args()
    
    args.show_layer = True
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    draw(args.draw_path)

