import os
from typing import Dict
import numpy as np

from yolox.utils import logger


def write_results(filename, results_dict: Dict, data_type: str):
    if not filename:
        return
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)

    if data_type in ('mot', 'mcmot', 'lab'):
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian -1 -1 -10 {x1} {y1} {x2} {y2} -1 -1 -1 -1000 -1000 -1000 -10 {score}\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, frame_data in results_dict.items():
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in frame_data:
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, score=1.0)
                f.write(line)
    logger.info('Save results to {}'.format(filename))


def read_results(filename, data_type: str, is_gt=False, is_ignore=False):
    if data_type in ('mot', 'lab'):
        read_fun = read_mot_results
    elif data_type =="mcmot":
        read_fun = read_mcmot_results
    else:
        raise ValueError('Unknown data type: {}'.format(data_type))

    return read_fun(filename, is_gt, is_ignore)


def read_mcmot_results(filename, is_gt, is_ignore):
    results_dict = dict()
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')
                if len(linelist) < 7:
                    continue
                fid = int(linelist[0])
                if fid < 1:
                    continue
                if is_gt:
                    score = 1
                elif is_ignore:
                    continue
                else:
                    score = float(linelist[6])
                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])
                cls = int(linelist[7])
                results_dict.setdefault(cls,dict())
                results_dict[cls].setdefault(fid, list())
                results_dict[cls][fid].append((tlwh, target_id, score))  
    return results_dict



def read_mot_results(filename, is_gt, is_ignore):
    valid_labels = {1,2,3,4,5,6,7,8}
    ignore_labels = {}
    results_dict = dict()
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')
                if len(linelist) < 7:
                    continue
                fid = int(linelist[0])
                if fid < 1:
                    continue
                results_dict.setdefault(fid, list())

                box_size = float(linelist[4]) * float(linelist[5])

                if is_gt:
                    score = 1
                elif is_ignore:
                    if 'MOT16-' in filename or 'MOT17-' in filename:
                        label = int(float(linelist[7]))
                        vis_ratio = float(linelist[8])
                        if label not in ignore_labels and vis_ratio >= 0:
                            continue
                    else:
                        continue
                    score = 1
                else:
                    score = float(linelist[6])
                
                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])
                
                results_dict[fid].append((tlwh, target_id, score))

    return results_dict


def unzip_objs(objs):
    if len(objs) > 0:
        tlwhs, ids, scores = zip(*objs)
    else:
        tlwhs, ids, scores = [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)

    return tlwhs, ids, scores

def unzip_objs_with_cls(objs):
    if len(objs) > 0:
        tlwhs, ids, scores,cls = zip(*objs)
    else:
        tlwhs, ids, scores,cls = [], [], [],[]
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)

    return tlwhs, ids, scores,cls