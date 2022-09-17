
import math
import os
from traceback import print_tb
import numpy as np
import copy
import motmetrics as mm
mm.lap.default_solver = 'lap'


def read_results(filename, is_gt, cls):

    results_dict = dict()
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')
                if cls != int(linelist[7]):
                    continue

                if len(linelist) < 7:
                    continue
                fid = int(linelist[0])
                if fid < 1:
                    continue
                results_dict.setdefault(fid, list())

                if is_gt:
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


class Evaluator(object):

    def __init__(self, data_root, seq_name, cls):
        self.data_root = data_root
        self.seq_name = seq_name
        self.cls = cls

        self.load_annotations()
        self.reset_accumulator()

    def load_annotations(self):

        gt_filename = os.path.join(
            self.data_root, self.seq_name, 'gt', 'gt.txt')
        self.gt_frame_dict = read_results(
            gt_filename, is_gt=True, cls=self.cls)

    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, rtn_events=False):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        # gts
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

        # get distance matrix
        iou_distance = mm.distances.iou_matrix(
            gt_tlwhs, trk_tlwhs, max_iou=0.5)

        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc, 'last_mot_events'):
            # only supported by https://github.com/longcw/py-motmetrics
            events = self.acc.last_mot_events
        else:
            events = None
        return events

    def eval_file(self, filename):
        self.reset_accumulator()

        result_frame_dict = read_results(filename, is_gt=False, cls=self.cls)
        frames = sorted(list(set(self.gt_frame_dict.keys()) | set(
            result_frame_dict.keys())))
        # frames = sorted(list(set(result_frame_dict.keys())))
        for frame_id in frames:
            trk_objs = result_frame_dict.get(frame_id, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            self.eval_frame(frame_id, trk_tlwhs, trk_ids, rtn_events=False)

        return self.acc

    @staticmethod
    def get_summary(accs, names, metrics=('mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall')):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True
        )

        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()


def calculate_S(mota, idf1):
    if mota < 0.0:
        print("warning, mota is lowwer than 0")
        return 0.0
    if mota == 0.0 and idf1 == 0.0:
        print("warning, mota or idf1 is 0")
        return 0.0
    return 2.0*mota*idf1/(mota+idf1)


def calculate_avg(evaluate_results):
    weight = [0.4, 0.1, 0.05, 0.15, 0.1, 0.15, 0.05]
    avg = 0.0
    for i, score in enumerate(evaluate_results):
        if math.isnan(score):
            print("warning, nan is appearing, Make sure that the GT of the test set contains at least the relevant category")
            avg += weight[i]*0
        else:
            avg += weight[i]*score
    evaluate_results.append(round(avg, 2))
    return evaluate_results


if __name__ == "__main__":

    data_root = "/home/lzq/Doc/Research/Dataset/MOT/jrb/images/test1"
    seqs_str = '2,18,22,28,32,41,45,46,55,62,69,80,87,89,90,101,102,103,105'
    result_root = "assets/w6-1920-motion-140/results"

    seqs = [seq.strip() for seq in seqs_str.split(',')]
    mcaccs = dict()
    evaluate_results = []
    for seq in seqs:
        result_filename = result_root + seq + '.txt'
        for cls in range(1, 8):
            mcaccs.setdefault(cls, list())
            evaluator = Evaluator(data_root, seq, cls)
            mcaccs[cls].append(evaluator.eval_file(result_filename))

    for cls in range(1, 8):
        metrics = mm.metrics.motchallenge_metrics
        mh = mm.metrics.create()
        summary = Evaluator.get_summary(mcaccs[cls], seqs, metrics)
        # strsummary = mm.io.render_summary(summary,formatters=mh.formatters,namemap=mm.io.motchallenge_metric_names)
        # print(strsummary)
        mota = summary["mota"]["OVERALL"]*100
        idf1 = summary["idf1"]["OVERALL"]*100

        if math.isnan(mota) or math.isnan(idf1):
            evaluate_results.append(calculate_S(mota, idf1))
        else:
            evaluate_results.append(round(calculate_S(mota, idf1), 2))

    evaluate_results_with_avg = calculate_avg(evaluate_results)
    # S of class 1,2,3,4,5,6,7 and all
    print("S of class 1,2,3,4,5,6,7 and all:")
    print(evaluate_results_with_avg)
