
import os
import sys
import cv2
from loguru import logger
import numpy as np
import torch
import torchvision

from utils.datasets import letterbox

def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))

def write_results_with_socre_and_cls(filename, results, data_type):
    if data_type == 'mot':
        # save_format = '{frame},{id},{x1},{y1},{w},{h},{score},{cls},-1,-1\n'
        # save_format = '{frame},{id},{x1},{y1},{w},{h},{score},{cls},-1\n'  # 符合杰瑞杯的形式
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,{cls},0\n'  # 符合杰瑞杯的形式
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids ,scores, classses in results:
            
            for tlwh, track_id ,score,cls,in zip(tlwhs, track_ids,scores,classses):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x1 , y1, w, h = round(x1,2) , round(y1,2), round(w,2), round(h,2) 
                x2, y2 = x1 + w, y1 + h
                score = round(score,2)
                # 网络预测0-7，实际结果1-8
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h,score=score,cls=cls+1)
                f.write(line)


def preproc(image, input_size, mean=None, std=None, swap=(2, 0, 1)):
    """
    :param image:
    :param input_size: (H, W)
    :param mean:
    :param std:
    :param swap:
    :return:
    """

    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0

    img = np.array(image)

    ## ----- Resize
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(img,
                             (int(img.shape[1] * r), int(img.shape[0] * r)),
                             interpolation=cv2.INTER_LINEAR).astype(np.float32)
    ## ----- Padding

    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    #  # lzq+ 改为对称填充
    # padded_img = letterbox(image, input_size,stride=64)[0]

    ## ----- BGR to RGB
    padded_img = padded_img[:, :, ::-1]

    ## ----- Normalize to [0, 1]
    padded_img /= 255.0

    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std

    ## ----- HWC ——> CHW
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

    return padded_img, r


def post_process(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    """
    :param prediction:
    :param num_classes:
    :param conf_thre:
    :param nms_thre:
    :return:
    """
    box_corner = prediction.new(prediction.shape)

    ## ----- cxcywh2xyxy
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2.0
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2.0
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2.0
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2.0
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        # nms_out_index = torchvision.ops.batched_nms(detections[:, :4],detections[:, 4] * detections[:, 5],detections[:, 6],nms_thre)
        # 使用nms时，不考虑类别，即使是不同类别也不能太接近，防止对目标预测不同类，导致多个轨迹  然而效果略有降低（0.01%）
        nms_out_index = torchvision.ops.nms(detections[:, :4],detections[:, 4] * detections[:, 5],nms_thre)

        detections = detections[nms_out_index]

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output
