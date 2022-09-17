import numpy as np

from utils.general import scale_coords_upper_left

from .matching import fuse_classes_width_height, iou_distance,linear_assignment,fuse_score, position_distance
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
 
    def __init__(self, tlwh, score,classes):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.classes = classes

    def activate(self, frame_id):
        """Start a new tracklet"""
        self.track_id = self.next_id()
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        # Assign the current detection directly to the strack 
        self._tlwh = new_track.tlwh
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        return self._tlwh.copy()


    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class NOKFTracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0
        self.args = args

        self.first_track_thresh = args.first_track_thresh
        self.second_track_thresh = args.second_track_thresh
        self.det_thresh = args.det_thresh

        self.first_match_thresh = args.first_match_thresh
        self.second_match_thresh = args.second_match_thresh
        self.motion_match_thresh = args.motion_match_thresh
        self.unconfirmed_match_thresh_iou = args.unconfirmed_match_thresh_iou
        self.unconfirmed_match_thresh_motion = args.unconfirmed_match_thresh_motion
        self.motion_thresh = args.motion_thresh

        self.use_motion = args.use_motion

        # self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.buffer_size = args.track_buffer
        self.max_time_lost = self.buffer_size


    def update_with_motion(self, output_results, img_info, img_size): #在第一帧中，没有做跟踪，只是更新了状态
        self.frame_id += 1
         # ----- reset the track ids in the first frame
        if self.frame_id == 1:
            STrack.init_id()

        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        elif output_results.shape[1] == 7:
            scale_coords_upper_left(img_size, output_results[:, :4], img_info)
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
            classes = output_results[:, 6]
        elif output_results.shape[1] == 6:
            scale_coords_upper_left(img_size, output_results[:, :4], img_info)
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            classes = output_results[:, 5]

        remain_inds = scores > self.first_track_thresh
        inds_low = scores > self.second_track_thresh
        inds_high = scores < self.first_track_thresh

        inds_second = np.logical_and(inds_low, inds_high)

        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        class_keep = classes[remain_inds]

        dets_second = bboxes[inds_second]
        scores_second = scores[inds_second]
        class_second = classes[inds_second]

        if len(dets) > 0:
            # Detections
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for (tlbr, s, c) in zip(dets, scores_keep, class_keep)] 
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes and IoU distance'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Non-KF do not predict
        # STrack.multi_predict(strack_pool)
        dists = iou_distance(strack_pool, detections)
        dists = fuse_classes_width_height(dists,strack_pool,detections) 
        dists = fuse_score(dists, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.first_match_thresh)
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes and IoU distance '''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s,c) for (tlbr, s,c) in zip(dets_second, scores_second,class_second)]
        else:
            detections_second = []
        # r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        r_tracked_stracks = [strack_pool[i] for i in u_track] 
        dists = iou_distance(r_tracked_stracks, detections_second)
        dists = fuse_classes_width_height(dists,r_tracked_stracks,detections_second) 
        
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=self.second_match_thresh)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 4: Third association, with high score detection boxes and Gaussian distance'''
        if self.use_motion:

            # Get the remain of the high score detection boxes
            detections = [detections[i] for i in u_detection]
            # Get the remain of stracks
            # r_r_tracked_stracks = [r_tracked_stracks[i] for i in u_track if r_tracked_stracks[i].state == TrackState.Tracked]
            r_r_tracked_stracks = [r_tracked_stracks[i] for i in u_track]
        
            # Get the Gaussian distance
            dists = position_distance(r_r_tracked_stracks, detections,motion_thresh=self.motion_thresh)
            dists = fuse_classes_width_height(dists,r_r_tracked_stracks,detections)  
            matches, u_track, u_detection = linear_assignment(dists, thresh=self.motion_match_thresh)
            for itracked, idet in matches:
                track = r_r_tracked_stracks[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_starcks.append(track)
                else:
                    # track.re_activate(det, self.frame_id, new_id=False)
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
            r_tracked_stracks = r_r_tracked_stracks 

        ''' Step 5: Mark the lost stracks  '''
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Step 6: Deal with unconfirmed tracks, usually tracks with only one beginning frame, using IoU distance'''
        detections = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)
        dists = fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=self.unconfirmed_match_thresh_iou)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])

        '''Step 7: Deal with unconfirmed tracks, using Gaussian distance'''
        if self.use_motion:
            detections = [detections[i] for i in u_detection]
            r_unconfirmed = [unconfirmed[i] for i in u_unconfirmed]
            dists = position_distance(r_unconfirmed, detections,motion_thresh=self.motion_thresh)
            dists = fuse_classes_width_height(dists,r_unconfirmed,detections) 
            matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=self.unconfirmed_match_thresh_motion)
            for itracked, idet in matches:
                r_unconfirmed[itracked].update(detections[idet], self.frame_id)
                activated_starcks.append(r_unconfirmed[itracked])
            unconfirmed = r_unconfirmed 
            
        '''Step 8: Remove unconfirmed stracks'''
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 9: Init new stracks, with high score detection boxes """
        for inew in u_detection:#
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.frame_id)
            activated_starcks.append(track)

        """ Step 10: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
