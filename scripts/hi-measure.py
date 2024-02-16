"""
@Time       : 2019/10/17 12:44
@Author     : Qin Dian
@Version    : 1.0.0
@Descrption : 统计
"""
import os
import numpy as np
import datetime
import argparse
from multiprocessing import Pool
from loguru import logger
from pathlib import Path
from skimage import measure
import traceback

ALL_CLASSED = 7
MAP3 = {0: 'All Tumor', 1: 'Liver', 2: 'Cancer', 4: 'Hem', 5: 'Cyst'} # 肿瘤，肝，恶性肿瘤，血管瘤, 囊肿


class Evaluation:
    def __init__(self):
        self.minimum = 1e-7
        self.scores = {}
        self.count = []
        self.call = np.zeros(ALL_CLASSED - 1)
        self.ac = np.zeros(ALL_CLASSED - 1)
        self.seg_tumors = np.zeros(ALL_CLASSED - 1)
        self.gt_tumors = np.zeros(ALL_CLASSED - 1)
        self.voe = False

    def evaluate(self):
        """
        多线程度量
        """
        self.count = np.zeros(ALL_CLASSED)

        for i in range(0, ALL_CLASSED):
            # 'dice0' represents all tumor
            self.scores['dice' + str(i)] = 0
            self.scores['voe' + str(i)] = 0

        if args.parallel:
            print(f'***Warning: multiprocessing model is on, thread number: {args.thread_num}')
            pool = Pool(args.thread_num)
            result = pool.map(self.run, os.listdir(args.pre_dir))
            pool.close()
            pool.join()

            for r in result:
                self.count += r[0]
                for i in range(0, ALL_CLASSED):
                    # 'dice0' represents all tumor
                    self.scores['dice' + str(i)] += r[1]['dice' + str(i)]
                self.gt_tumors += r[2]
                self.call += r[3]
                self.seg_tumors += r[4]
                self.ac += r[5]
        else:
            print('***Warning: multiprocessing model is off')
            for file in os.listdir(args.pre_dir):
                self.run(file)

        out = self._calculate(args.num_classes)
        return out
        # today = datetime.date.today().strftime('%y%m%d')
        # if algorithm_name is None:
        #     return out
        # else:
        #     return f'{algorithm_name}|{version}|{variation}|{out}{today}|{modifier}|TBD'

    def run(self, file):
        """
        单个线程核心函数
        """
        out = ''
        series_num = args.series_num
        pre_img = self._load(os.path.join(args.pre_dir, file))
        gt_img = self._load(os.path.join(args.gt_dir, file))

        if series_num is not None:
            pre_img = pre_img[series_num]
            gt_img = gt_img[series_num]

        if args.num_classes == 3:
            gt_img[(gt_img == 6) | (gt_img == 3)] = 2
            pre_img[(pre_img == 6) | (pre_img == 3)] = 2

        out += self._measure_by_class(pre_img, gt_img, ALL_CLASSED, file)
        out += self._measure_by_tumor(pre_img, gt_img, series_num, file)
        
        print(f'{str(file)} done')
        # multiprocess needs return
        return self.count, self.scores, self.gt_tumors, self.call, self.seg_tumors, self.ac

    def _measure_by_class(self, pre_img, gt_img, num_class, file):
        """
        Dice metric function, to measure overlap
        :param pre_img:  segmentation matrix
        :param gt_img:  ground truth matrix
        :param num_class: class number
        :return:
        """
        out = ''
        for t_class in range(num_class):
            if t_class == 0:
                seg = pre_img >= 2
                gt = gt_img >= 2
            elif t_class == 1:
                seg = pre_img >= t_class
                gt = gt_img >= t_class
            else:
                seg = pre_img == t_class
                gt = gt_img == t_class
            x = np.sum(seg)
            y = np.sum(gt)

            intersection = np.sum(seg * gt)

            ###################
            if t_class not in MAP3.keys():
                continue

            if t_class == 0:
                dice_score = self._dice(x, y, intersection)
                voe_score = self._voe(x, y, x - intersection)
            else:
                if y == 0:
                    if x == 0:
                        dice_score = 1.
                        voe_score = 0.
                    else:
                        dice_score = 0
                        voe_score = 1
                else:
                    dice_score = self._dice(x, y, intersection)
                    voe_score = self._voe(x, y, x - intersection)
            self.count[t_class] += 1
            ####################

            # if x == 0 or y == 0:
            #     continue
            # else:
            #     self.count[t_class] += 1
            #
            # intersection = np.sum(seg * gt)
            # dice_score = self._dice(x, y, intersection)
            # voe_score = self._voe(x, y, x - intersection)

            self.scores['dice' + str(t_class)] += dice_score
            self.scores['voe' + str(t_class)] += voe_score
            out += f'{file}_class{t_class} {MAP3[t_class].ljust(10)}: dice|voe: {dice_score}|{voe_score}\n'
        return out

    def _measure_by_tumor(self, seg_img, gt_img, series_num, file):
        """
        按照肿瘤个数测算准确率和召回率
        :param pre_img:  segmentation matrix
        :param gt_img:  ground truth matrix
        :param num_class: class number
        """
        out = ''
        for i in range(1, 7):
            if i == 1:
                # i==1 means all tumor(segmentation without classification)
                seg = (seg_img > i).astype(np.uint8)
                gt = (gt_img > i).astype(np.uint8)
            else:
                seg = (seg_img == i).astype(np.uint8)
                gt = (gt_img == i).astype(np.uint8)

            tumor_num_r = tumor_num_p = get_r = get_p = 0
            if series_num is not None:
                # lr
                tumor_num_r, get_r = self._lr_lp(gt, seg)
                # lp
                tumor_num_p, get_p = self._lr_lp(seg, gt)
            else:
                for s in range(seg.shape[0]):
                    # lr
                    ttr, tgt = self._lr_lp(gt[s], seg[s])
                    tumor_num_r += ttr
                    get_r += tgt
                    # lp
                    ttp, tgp = self._lr_lp(seg[s], gt[s])
                    tumor_num_p += ttp
                    get_p += tgp
            self.gt_tumors[i - 1] += tumor_num_r
            self.call[i - 1] += get_r
            self.seg_tumors[i - 1] += tumor_num_p
            self.ac[i - 1] += get_p

            if i == 1:
                if tumor_num_r == 0 and tumor_num_p == 0:
                    recall = pre = 1
                elif tumor_num_p == 0 or tumor_num_r == 0:
                    recall = pre = 0
                else:
                    recall = get_r / (tumor_num_r + self.minimum)
                    pre = get_p / (tumor_num_p + self.minimum)

                out += f'{file}_TSRecall:    {recall}'
                out += '\n'
                out += f'{file}_TSPrecision: {pre}'
        return out

    @staticmethod
    def _lr_lp(base, pool):
        """
        计算准确率和召回率
        calculate lr: base=gt, pool=seg
        calculate lp: base=set, pool=gt
        :param base: base map
        :param pool: pool map
        :return: base中的病灶数, base中能和pool匹配的病灶数
        """
        tumor_num = 0
        get = 0
        if np.any(base):
            tumors = measure.label(base, connectivity=1)
            num = tumors.max()
            tumor_num += num
            for j in range(1, num + 1):
                if np.any((tumors == j) * pool):
                    get += 1
        return tumor_num, get

    def _dice(self, x, y, intersection):
        """
        计算Dice
        """
        if x == y == 0:
            return 1.
        score = 2 * intersection / (x + y + self.minimum)
        return score
    
    def _f2_score(self, precision, recall):
        """
        计算f2 score
        """
        score = 5 * precision * recall / (4*precision + recall + self.minimum)
        return score

    def _voe(self, x, y, differ):
        """
        计算voe
        """
        score = 2 * differ / (x + y + self.minimum)
        return score

    def _rvd(self, x, y):
        """
        计算rvd
        """
        score = x / (y + self.minimum) - 1
        return score

    def _calculate(self, num_classes):
        liver = ''
        tumor = ''
        dice_all = ''
        voe_all = ''
        tcr = ''
        tcp = ''
        d = 4
        for i, (key, value) in enumerate(self.scores.items()):
            count = self.count[int(key[-1])]
            if self.voe is False and 'voe' in key:
                continue

            if '0' in key:
                tumor += ('0|' if count == 0 else str(round(value / count, d)) + '|')
            elif '1' in key:
                liver += ('0|' if count == 0 else str(round(value / count, d)) + '|')
            elif 'dice' in key:
                if num_classes == 3:
                    if ('3' in key) or ('6' in key):
                        continue
                dice_all += ('0, ' if count == 0 else str(round(value / count, d)) + ', ')
            elif 'voe' in key:
                voe_all += ('0, ' if count == 0 else str(round(value / count, d)) + ', ')

        tsr = str(round(self.call[0] / self.gt_tumors[0], d))
        tsp = str(round(self.ac[0] / self.seg_tumors[0], d))
        for i in range(1, len(self.call)):
            if num_classes == 3:
                if i == 2 or i == 5:
                    continue
            tcr += ('-' if self.gt_tumors[i] == 0 else str(round(self.call[i] / self.gt_tumors[i], d))) + ', '
            tcp += ('-' if self.seg_tumors[i] == 0 else str(round(self.ac[i] / self.seg_tumors[i], d))) + ', '

        if self.voe is False:
            result = liver + tumor + dice_all[:-2] + '|' + tsr + '|' + tsp + '|' + tcr[:-2] + '|' + tcp[:-2]
        else:
            result = liver + tumor + dice_all[:-2] + '|' + voe_all[:-2] + '|' + tsr + '|' + tsp + '|'
        result += f'| f2={self._f2_score(self.ac[0] / self.seg_tumors[0], self.call[0] / self.gt_tumors[0]):.4f}'
        return result

    @staticmethod
    def _load(path):
        """
        内部NPZ文件加载方法
        """
        file = np.load(path)
        liver_mask, tumor_mask = file.get("liver_mask"), file.get("tumor_mask")
        # img = np.array(file.get('arrays', file.get('arr_0')))
        liver_mask[tumor_mask > 0] = tumor_mask[tumor_mask > 0]
        # liver_mask = liver_mask.reshape(len(liver_mask), -1, 512, 512)
        # img[img > 1] = img[img > 1] >> 1
        return liver_mask


def main():
    e = Evaluation()
    logger.info(f'{args.pre_dir}: {e.evaluate()}')


if __name__ == '__main__':
    # 脚本参数定义
    parser = argparse.ArgumentParser(description="hi_measure")
    parser.add_argument('--output_dir', type=str, help='output root dir')
    parser.add_argument('--pre_dir', type=str, default='/data0/wulei/train_log/segmentor_dual_p_d/pred-28-post')
    parser.add_argument('--gt_dir', type=str, default="/data4/liver_CT4_Z2_ts2.2")
    parser.add_argument('--num_classes', type=int, default=3, choices=(3, 5))
    parser.add_argument('--series_num', type=int, default=2, choices=(0, 1, 2, 3))
    parser.add_argument('--parallel', action='store_true', help='多进程模式，默认开启')
    parser.add_argument('--thread_num', type=int, default=20, help='只有启动多进程模式（parallel为True）才有效')

    args = parser.parse_args()

    if args.output_dir:
        args.output_dir = Path(args.output_dir)
        logger.add(args.output_dir / 'test_log.txt')
    
    main()
