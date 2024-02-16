import os
import cv2
import numpy as np


COLOR_TABLE = [(), (), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 97, 0)]     # set color of different tumor BGR
CLS_LIS = [2, 3, 4]         # set index of tumor to display


def display(img, mask, name, save_path):
    """
    画图
    :param img: ct矩阵
    :param mask: mask矩阵
    :param name: 存储名
    """
    # shape [s, n, h, w]
    # img, mask = get_img(name)
    assert len(img) == len(mask)
    result = []
    
    img = np.clip(img, -60, 160)
    img = (img + 60.) / (160. + 60.)

    mask[(mask == 2) | (mask == 3) | (mask == 6) | (mask == 7)] = 2
    mask[mask == 4] = 3
    mask[mask == 5] = 4

    for img_single, mask_single in zip(img, mask):
        single_series = []
        for i in range(len(img_single)):
            img_slice, mask_slice = img_single[i], mask_single[i]
            img_slice = img_slice * 255
            img_slice = cv2.cvtColor(img_slice.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            for cls in CLS_LIS:
                mask_slice_c = np.zeros_like(mask_slice)
                mask_slice_c[mask_slice == cls] = 1
                cout, _ = cv2.findContours(mask_slice_c.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_slice, cout, -1, COLOR_TABLE[cls], thickness=2)

            single_series.append(img_slice)

        result.append(cv2.vconcat(single_series))

    result = cv2.hconcat(result)
    cv2.imwrite(os.path.join(save_path, name+".png"), result, [cv2.IMWRITE_PNG_COMPRESSION, 10])
