import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import random

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from bmaskrcnn import add_boundary_preserving_config

im_folder = './datasets/JPEGImages/'
save_folder = './result_struc'

for im_file in os.listdir(im_folder):
    im = cv2.imread(os.path.join(im_folder, im_file))

    save_result_path = os.path.join(save_folder, im_file)
    txt_file_path = os.path.join(save_folder, im_file.replace('.jpg', '.txt'))  # 生成对应的txt文件路径

    height = im.shape[0]
    width = im.shape[1]
    dpi = 500

    cfg = get_cfg()
    add_boundary_preserving_config(cfg)
    cfg.merge_from_file('./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  #模型阈值
    cfg.MODEL.WEIGHTS = './datasets/model_final.pth'
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    pred_classes = outputs["instances"].pred_classes
    pred_boxes = outputs["instances"].pred_boxes

    # 将框的信息保存到txt文件
    with open(txt_file_path, 'w') as f:
        for i in range(len(pred_classes)):
            # 获取预测框的坐标，按格式输出
            box = pred_boxes[i].tensor.cpu().numpy()[0]  # 获取第i个框的坐标 [x_min, y_min, x_max, y_max]
            class_id = pred_classes[i].item()  # 获取类别ID

            # 归一化框的坐标
            x_min_norm = box[0] / width
            y_min_norm = box[1] / height
            x_max_norm = box[2] / width
            y_max_norm = box[3] / height

            # 格式化为：class_id, x_min, y_min, x_max, y_max
            line = f"{class_id} {x_min_norm} {y_min_norm} {x_max_norm} {y_max_norm}\n"
            f.write(line)

    # 在原图上画出检测结果
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # 保存图片
    plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.imshow(v.get_image())
    plt.savefig(save_result_path)  # 保存检测结果图像
