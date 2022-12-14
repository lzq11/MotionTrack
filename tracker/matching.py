import lap
import numpy as np
import scipy
from cython_bbox import bbox_overlaps as bbox_ious
from scipy.spatial.distance import cdist

from .kalman_filter import chi2inv95


def merge_matches(m1, m2, shape):
    """
    :param m1:
    :param m2:
    :param shape:
    :return:
    """
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    """
    :param cost_matrix:
    :param thresh:
    :return:
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def iou_distance_with_alias(atracks, btracks,alias):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    alias = np.array([alias[0],alias[1],alias[0],alias[1]])
    atlbrs = [tlbr + alias for tlbr in atlbrs]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Normalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    """
    :param kf:
    :param cost_matrix:
    :param tracks:
    :param detections:
    :param only_position:
    :return:
    """
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    """
    :param kf:
    :param cost_matrix:
    :param tracks:
    :param detections:
    :param only_position:
    :param lambda_:
    :return:
    """
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    """
    :param cost_matrix:
    :param tracks:
    :param detections:
    :return:
    """
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    # fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    """
    :param cost_matrix:
    :param detections:
    :return:
    """
    if cost_matrix.size == 0:
        return cost_matrix

    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim

    return fuse_cost


def fuse_costs(cost_mat1, cost_mat2):
    """
    :param cost_mat1:
    :param cost_mat2:
    :return:
    """
    if cost_mat1.size == 0:
        return cost_mat1
    if cost_mat2.size == 0:
        return cost_mat2

    sim1 = 1.0 - cost_mat1
    sim2 = 1.0 - cost_mat2

    fuse_sim = sim1 * sim2
    fuse_cost = 1.0 - fuse_sim

    return fuse_cost


def weight_sum_costs(cost_mat1, cost_mat2, alpha=0.5):
    """
    :param cost_mat1:
    :param cost_mat2:
    :param alpha:
    :return:
    """
    if cost_mat1.size == 0:
        return cost_mat1
    if cost_mat2.size == 0:
        return cost_mat2

    sim1 = 1.0 - cost_mat1
    sim2 = 1.0 - cost_mat2

    fuse_sim = sim1 * alpha + (1.0 - alpha) * sim2
    fuse_cost = 1.0 - fuse_sim

    return fuse_cost

def fuse_classes(cost_matrix, strack_pool,detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix

    strack_classes = np.array([strack.classes for strack in strack_pool])
    det_classes = np.array([det.classes for det in detections])

    class_matrix = iou_sim.copy()
    for i in range(strack_classes.size):
        for j in range(det_classes.size):
            if strack_classes[i] == det_classes[j]:
                class_matrix[i,j] = 1
            else:
                class_matrix[i,j]=0
    fuse_sim = iou_sim * class_matrix
    fuse_cost = 1 - fuse_sim
    return fuse_cost

def fuse_classes_width_height(cost_matrix, strack_pool,detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix

    strack_classes = np.array([strack.classes for strack in strack_pool])
    det_classes = np.array([det.classes for det in detections])

    strack_tlwh = np.array([strack.tlwh for strack in strack_pool])
    det_tlwh = np.array([det.tlwh for det in detections])          

    classes_width_height_matrix = iou_sim.copy()
    for i in range(strack_classes.size):
        for j in range(det_classes.size):
            if strack_classes[i] == det_classes[j] and (1/4 < (strack_tlwh[i,2]*strack_tlwh[i,3])/(det_tlwh[j,2]*det_tlwh[j,3]) < 4):
                classes_width_height_matrix[i,j] = 1
            else:
                classes_width_height_matrix[i,j]=0

    fuse_sim = iou_sim * classes_width_height_matrix
    fuse_cost = 1 - fuse_sim
    return fuse_cost

def motion_distance(kf,tracks, detections,only_position=False, motion_thresh=100):

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if len(detections) != 0 :
        gating_threshold = motion_thresh*motion_thresh
        measurements = np.asarray([det.to_xyah() for det in detections])
        for row, track in enumerate(tracks):
            gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position, metric='gaussian')
            cost_matrix[row] = gating_distance/gating_threshold
            cost_matrix[row, gating_distance > gating_threshold] = 1
    return cost_matrix

def position_distance(tracks, detections,motion_thresh=100):

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if len(detections) != 0 :
        gating_threshold = motion_thresh*motion_thresh
        measurements = np.asarray([det.to_xyah() for det in detections])
        for row, track in enumerate(tracks):
            mean = track.to_xyah()
            d = measurements - mean
            gating_distance = np.sum(d * d, axis=1)
            cost_matrix[row] = gating_distance/gating_threshold
            cost_matrix[row, gating_distance > gating_threshold] = 1
    return cost_matrix
    