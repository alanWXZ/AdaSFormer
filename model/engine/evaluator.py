import os
import numpy as np
import time
from tqdm import tqdm

import torch
import re

from engine.logger import get_logger
from utils.pyt_utils import load_model, link_file, ensure_dir

from engine.sscMetrics import SSCMetrics

logger = get_logger()


def vol2points_numba(pred,gt):
    # 颜色表
    colorMap = np.array([[22, 191, 206], [214, 38, 40], [43, 160, 43], [158, 216, 229],
                         [114, 158, 206], [204, 204, 91], [255, 186, 119], [147, 102, 188],
                         [30, 119, 181], [188, 188, 33], [255, 127, 12], [196, 175, 214],
                         [153, 153, 153]], dtype=np.int32)

    # 先统计非 307200 的点数量
    count = 0
    for x in range(gt.shape[0]):
        for y in range(gt.shape[1]):
            for z in range(gt.shape[2]):
                if pred[x, y, z] > 0 and gt[x][y][z] !=0 and gt[x][y][z] !=255 and pred[x][y][z]<255:
                    count += 1

    points = np.zeros((count, 3), dtype=np.int32)
    rgb = np.zeros((count, 3), dtype=np.int32)

    idx = 0
    for x in range(gt.shape[0]):
        for y in range(gt.shape[1]):
            for z in range(gt.shape[2]):
                if pred[x, y, z] > 0 and gt[x][y][z] !=0 and gt[x][y][z] !=255 and pred[x][y][z]<255 :
                    points[idx, 0] = x
                    points[idx, 1] = y
                    points[idx, 2] = z
                    rgb[idx, :] = colorMap[pred[x][y][z]]
                    idx += 1

    return points, rgb
def writeply(filename, points, rgb):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {points.shape[0]}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for i in range(points.shape[0]):
            f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} {rgb[i, 0]} {rgb[i, 1]} {rgb[i, 2]}\n")
class Evaluator(object):
    def __init__(self, dataset, class_num, image_mean, image_std, network,
                 multi_scales, is_flip, devices,
                 verbose=False, save_path=None, show_image=False):
        self.eval_time = 0
        self.dataset = dataset
        self.ndata = len(dataset._file_names)
        
        self.class_num = class_num
        self.image_mean = image_mean
        self.image_std = image_std
        self.multi_scales = multi_scales
        self.is_flip = is_flip
        self.network = network
        self.devices = devices

        self.val_func = None

        self.verbose = verbose
        self.save_path = save_path
        if save_path is not None:
            ensure_dir(save_path)
        self.show_image = show_image
        self.val_metrics = SSCMetrics(12)

    def run(self, model_path, model_indice, log_file, log_file_link):
        """There are four evaluation modes:
            1.only eval a .pth model: -e *.pth
            2.only eval a certain epoch: -e epoch
            3.eval all epochs in a given section: -e start_epoch-end_epoch
            4.eval all epochs from a certain started epoch: -e start_epoch-
            """
        if '.pth' in model_indice:
            models = [model_indice, ]
        elif "-" in model_indice:
            start_epoch = int(model_indice.split("-")[0])
            end_epoch = model_indice.split("-")[1]

            models = os.listdir(model_path)
            models.remove("epoch-last.pth")
            sorted_models = [None] * len(models)
            model_idx = [0] * len(models)

            for idx, m in enumerate(models):
                num = m.split(".")[0].split("-")[1]
                model_idx[idx] = num
                sorted_models[idx] = m
            model_idx = np.array([int(i) for i in model_idx])

            down_bound = model_idx >= start_epoch
            up_bound = [True] * len(sorted_models)
            if end_epoch:
                end_epoch = int(end_epoch)
                assert start_epoch < end_epoch
                up_bound = model_idx <= end_epoch
            bound = up_bound * down_bound
            model_slice = np.array(sorted_models)[bound]
            models = [os.path.join(model_path, model) for model in
                      model_slice]
        else:
            if os.path.exists(model_path):
                models = [os.path.join(model_path,
                                       'epoch-%s.pth' % model_indice), ]
            else:
                models = [None]

        results = open(log_file, 'a')
        link_file(log_file, log_file_link)

        for model in models:
            logger.info("Load Model: %s" % model)
            self.val_func = load_model(self.network, model)
            match = re.search(r'epoch-(\d+)', model)
            if match:
                epoch_num = int(match.group(1))  # 55
                epoch = f"{epoch_num}"

            if len(self.devices ) == 1:
                result_line = self.single_process_evalutation(epoch)

            results.write('Model: ' + model + '\n')
            #results.write(result_line)
            results.write('\n')
            results.flush()

        results.close()

    def single_process_evalutation(self, epoch):
        start_eval_time = time.perf_counter()

        logger.info(
                'GPU %s handle %d data.' % (self.devices[0], self.ndata))
        all_results = []

        for idx in tqdm(range(self.ndata)):

            dd = self.dataset[idx]


            results_dict = self.func_per_iteration(dd,self.devices[0])
            
            path = results_dict['name']
            new_path = path.replace('/media/psdz/data/dataset/occscannet','')
            new_path = new_path.replace('.pkl','')
            dir = '/media/psdz/data/dataset/occ_asformer/'
            points, rgb = vol2points_numba(results_dict['pred'],results_dict['label'])

            os.makedirs(os.path.dirname(dir + new_path), exist_ok=True)
            writeply(os.path.join(dir + new_path + '.ply'), points, rgb)


            
            self.val_metrics.add_batch(results_dict['pred'],results_dict['label'])
            #all_results.append(results_dict)
            
            
        stats = self.val_metrics.get_stats()
        result_line, score_ssc = self.compute_metric(all_results)
        #                         self.val_metrics
        print(stats)
        a= stats['iou_ssc_mean']
        self.val_metrics.reset()
        self.log_epoch_score(epoch, a)
        logger.info(
            'Evaluation Elapsed Time: %.2fs' % (
                    time.perf_counter() - start_eval_time))
        return

    def log_epoch_score(self, epoch_str: str, ssc_score: float, filename: str = "log.txt"):
        # 获取当前运行目录
        current_dir = os.getcwd()
        log_path = os.path.join(current_dir, filename)

        # 追加写入
        with open(log_path, 'a') as f:
            f.write(f"{epoch_str} {ssc_score:.6f} ")  # 保留4位小数

    def func_per_iteration(self, data, device):
        raise NotImplementedError

    def compute_metric(self, results):
        raise NotImplementedError

    def process_image_rgbd(self, img, disp, crop_size=None):
        img = img.cpu().numpy()
        p_img = img
        if disp is not None:
            p_disp = disp.cpu().numpy()
        else:
            p_disp = None
        return p_img, p_disp
