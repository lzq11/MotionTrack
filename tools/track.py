import sys
sys.path.append('./')

import time
import motmetrics as mm
from tracker.utils.datasets import LoadPaths
from tracker.utils.logger import setup_logger
import torch.backends.cudnn as cudnn
from tracker.utils.visualization import plot_tracking, plot_tracking_mc, plot_tracking_with_class
from loguru import logger
from tracker.utils.utils import post_process, preproc, write_results_with_socre_and_cls
import cv2
from tracker.nokalman_tracker import NOKFTracker
from tracker.motion_tracker import MotionTracker
from tracker.utils.timer import Timer
from collections import defaultdict
from utils.torch_utils import select_device, TracedModel
from utils.general import check_file, increment_path, non_max_suppression
from utils.datasets import letterbox
from tracker.utils.evaluation import Evaluator
from models.experimental import attempt_load
import argparse
import configparser
from logging import raiseExceptions

import os
from pathlib import Path

import numpy as np
import torch


# tracking
class Predictor(object):
    def __init__(self, model, exp, decoder=None, device="cpu", fp16=False, reid=False):
        """
        :param model:
        :param exp:
        :param trt_file:
        :param decoder:
        :param device:
        :param fp16:
        :param reid:
        """
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.conf_thresh = exp.conf_thres
        self.nms_thresh = exp.iou_thres
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.reid = reid
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        """
        :param img:
        :param timer:
        :return:
        """
        img_info = {"id": 0}

        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        # img, ratio = preproc(img, self.test_size, self.mean, self.std)
        # yolov7-w6 没有均值方差归一化，值都是在 0-1
        img, ratio = preproc(img, self.test_size)

        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()

        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()

            ## ----- forward
            outputs = self.model.forward(img)

            outputs, feature_map = outputs[0], outputs[1]       # # 128520,12

            outputs = post_process(
                outputs, self.num_classes, self.conf_thresh, self.nms_thresh)
            # outputs = non_max_suppression(outputs, self.conf_thresh, self.nms_thresh,multi_label=True)
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))

        return outputs, img_info


def eval_seq(args, exp, predictor, model, dataloader, data_type, result_filename, save_dir=None, show_image=False, frame_rate=30,):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    tracker = MotionTracker(args)
    # tracker = NOKFTracker(args)
    timer = Timer()
    results = []
    frame_id = 0
    for path in dataloader:

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(
                frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        if args.reid:
            outputs, feature_map, img_info = predictor.inference(path, timer)
        else:
            outputs, img_info = predictor.inference(path, timer)
        dets = outputs[0]

        online_tlwhs = []
        online_ids = []
        online_scores = []
        online_classids = []

        if dets is not None:
            # ----- update the frame
            img_size = [img_info['height'], img_info['width']]
            # online_targets = tracker.update(dets, img_size, exp.test_size)
            online_targets = tracker.update_with_motion(dets, img_size, exp.test_size)

            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                if tlwh[2] * tlwh[3] > 0:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_classids.append(int(t.classes))
            # save results
            results.append((frame_id+1, online_tlwhs, online_ids,
                           online_scores, online_classids))
            timer.toc()
        else:
            timer.toc()
            # online_im = img_info['raw_img']

        if show_image or save_dir is not None:
            # online_im = plot_tracking(img_info['raw_img'],online_tlwhs,online_ids,frame_id=frame_id + 1,fps=1.0 / timer.average_time)
            online_im = plot_tracking_with_class(
                img_info['raw_img'], online_tlwhs, online_ids, online_classids, args.class_names, frame_id=frame_id + 1, fps=1.0 / timer.average_time)

        if save_dir is not None:
            cv2.imwrite(os.path.join(
                save_dir, '{:05d}.jpg'.format(frame_id + 1)), online_im)
        frame_id += 1
    # save results
    write_results_with_socre_and_cls(result_filename, results, data_type)
    logger.info('save results to {}'.format(result_filename))

    return frame_id, timer.average_time, timer.calls


def track(opt, weights=None, batch_size=1,
          model=None, dataloader=None,
          save_dir=Path(''),  # for saving images
          save_txt=False,  # for auto-labelling
          trace=False,):

    # data_root = "/home/lzq/Doc/Research/Dataset/MOT/jrb/images/test1"
    data_root = opt.data_root

    seqs_str = '2,18,22,28,32,41,45,46,55,62,69,80,87,89,90,101,102,103,105'
    seqs = [seq.strip() for seq in seqs_str.split(',')]
    # seqs = sorted(os.listdir(opt.data_root))

    save_images = False
    save_videos = False
    show_image = False
    rank = 0
    class_names = opt.class_names.split(",")
    opt.class_names = class_names
    device = select_device(opt.device, batch_size=batch_size)
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name,
                    exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,exist_ok=True)  # make dir

    result_root = os.path.join(save_dir, 'results')
    visulation_root = os.path.join(save_dir, 'vis')
    os.makedirs(result_root, exist_ok=True)
    os.makedirs(visulation_root, exist_ok=True)
    setup_logger(save_dir, distributed_rank=rank, filename="val_log.txt", mode="a")
    cudnn.benchmark = True
    logger.info("Args: {}".format(opt))

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model

    if trace:
        model = TracedModel(model, device, opt.test_size)

    # Half
    # half precision only supported on CUDA
    opt.fp16 = device.type != 'cpu' and opt.fp16

    if opt.fp16:
        model.half()

    # Configure
    model.eval()
    exp = opt
    # ---------- Define the predictor
    predictor = Predictor(model, exp, device="gpu",fp16=opt.fp16, reid=opt.reid)
    data_type = "mot"

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        # 获取序列的fps
        seqinfoini = configparser.ConfigParser()
        seqinfoini.read(os.path.join(data_root, seq, "seqinfo.ini"))
        frame_rate = int(seqinfoini.get("Sequence", "frameRate"))  # 30

        # 如果要保存跟踪的图片结果或视频，则设置保存路径
        output_dir = os.path.join(
            visulation_root, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))

        dataloader = LoadPaths(os.path.join(data_root, seq, seq))
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))

        nf, ta, tc = eval_seq(opt, exp, predictor, model, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))

        if save_videos:
            video_path = os.path.join(save_dir, "vis_video")
            os.makedirs(video_path, exist_ok=True)
            output_video_path = os.path.join(video_path, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(
                output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(
        all_time, 1.0 / avg_time))
    # get total summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    print(strsummary)
    logger.info(strsummary)

    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='track.py')
    parser.add_argument('--weights', nargs='+', type=str,
                        default='runs/train/yolov7-w6/weights/epoch_099.pt',
                        help='model.pt path(s)')
    parser.add_argument('--data', type=str,
                        default='data/jmt2022.yaml',
                        help='*.data path')
    parser.add_argument('--batch-size', type=int,
                        default=1, help='size of each image batch')
    parser.add_argument('--test-size', type=int,
                        default=[1088, 1920],
                        # default=[512,960],
                        # default=[256,640],
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.1,
                        help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float,
                        default=0.65,
                        help='IOU threshold for NMS')
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--verbose', action='store_true',
                        help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true',
                        help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--project', default='runs/track',
                        help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true',
                        help='don`t trace model')

    # tracking args
    parser.add_argument("--num_classes", type=int, default=7, help="")
    parser.add_argument("--class_names", type=str,
                        default="sailing_boat,fishing_boat,floater,passenger_ship,speedboat,cargo,special_ship", help="")
    parser.add_argument("--first_track_thresh", type=float,
                        default=0.6, help="high detection threshold")
    parser.add_argument("--second_track_thresh", type=float,
                        default=0.1, help="low detection threshold")
    parser.add_argument("--det_thresh", type=float,
                        default=0.7, help="tracking threshold")
    parser.add_argument("--first_match_thresh", type=float,
                        default=0.98, help="high detection using IoU distance")
    parser.add_argument("--second_match_thresh", type=float,
                        default=0.98, help="low detection using IoU distance")
    parser.add_argument("--motion_match_thresh", type=float,
                        default=0.98, help="high detection using Gaussian distance")
    parser.add_argument("--unconfirmed_match_thresh_iou",
                        type=float, default=0.98, help="unconfirmed detection using IoU distance")
    parser.add_argument("--unconfirmed_match_thresh_motion",
                        type=float, default=0.98, help="unconfirmed detection using Gaussian distance")
    parser.add_argument("--track_buffer", type=int,
                        default=30, help="trajectory retention frame")
    parser.add_argument("--motion_thresh", type=int,
                        default=140, help="Gaussian distance threshold")
    parser.add_argument("--use_motion", type=bool,
                        default=True, help="whether to use Gaussian distance")
    parser.add_argument("--data_root", type=str,
                        default="/home/lzq/Doc/Research/Dataset/MOT/jrb/images/test1")
    parser.add_argument("--min-box-area", type=float,
                        default=0, help='filter out tiny boxes')
    parser.add_argument("--reid", type=bool,
                        default=False, help="True | False")
    parser.add_argument("--fp16", dest="fp16", default=True,
                        action="store_true", help="Adopting mix precision evaluating.")

    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    print(opt)

    track(opt, opt.weights, opt.batch_size, save_txt=opt.save_txt |
          opt.save_hybrid, trace=not opt.no_trace)
