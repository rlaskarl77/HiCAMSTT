from collections import OrderedDict
from collections import deque
from typing import List

import numpy as np

from tracking import matching
from tracking.kalman_filter import KalmanFilter


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, xy, xy_prev, score, buffer_size=30):

        # wait activate
        self._xy = xy
        self._xy_prev = xy_prev
        self.kalman_filter: KalmanFilter = None
        self.mean: np.ndarray = None
        self.covariance: np.ndarray = None
        self.is_activated: bool = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        # self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks: List["STrack"]):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
                
                stracks[i]._xy_prev = stracks[i]._xy
                stracks[i]._xy = mean[:2]

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.xy)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
            # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False, update_feature=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xy
        )

        # self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        if update_feature:
            self.update_features(new_track.curr_feat)

    def update(self, new_track: "STrack", frame_id, update_feature=False):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_track.xy)
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    def xy(self) -> np.ndarray:
        # if self.state == TrackState.Lost:
        #     return self.mean[:2]
        if self.mean is None:
            return self._xy
        return self.mean[:2]

    @property
    def xy_prev(self):
        return self._xy_prev

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker:
    def __init__(self,
                 conf_thres=0.1, 
                 track_buffer=5,
                 lapjv_thresh=0.25,
                 lapjv_thresh2=0.5,
                 max_spatial_dist=75.,
                 max_spatial_dist2=100.,
                 dist_alpha=0.5,
                 use_reid_tracking=True,
                 temp_mixing=True,
                 lambda_1=1.,
                 lambda_2=1.
                 ):
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0
        self.det_thresh = conf_thres
        self.max_time_lost = track_buffer

        self.kalman_filter = KalmanFilter()
        
        self.dist_alpha = dist_alpha
        
        self.lapjv_thresh = lapjv_thresh
        self.lapjv_thresh2 = lapjv_thresh2
        
        self.max_spatial_dist = max_spatial_dist
        self.max_spatial_dist2 = max_spatial_dist2
        
        self.reid = use_reid_tracking
        self.use_temporal_mixing = temp_mixing
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2


    def update(self, dets, dets_prev, score, reid=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        missed_stracks = []
        removed_stracks = []

        remain_inds = score > self.det_thresh - 0.1
        dets = dets[remain_inds]
        dets_prev = dets_prev[remain_inds]
        
        if self.reid:
            reid = reid[remain_inds]
            assert len(dets) == len(reid), '{} != {}'.format(len(dets), len(reid))

        if len(dets) > 0:
            """Detections"""
            detections: List[STrack] = [STrack(xy, xy_prev, s, self.max_time_lost) for
                          (xy, xy_prev, s) in zip(dets, dets_prev, score)]
            
            if self.reid:
                """Update reid features"""
                for track, feat in zip(detections, reid):
                    track.update_features(feat)
        else:
            detections: List[STrack] = []
        
        unconfirmed: List[STrack] = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        strack_pool: List[STrack] = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        

        ''' Step 2:association'''
        '''
            Step 2.1: Backward prediction with activated tracks
            Step 2.2: Forward prediction with lost + unmatched tracks
        '''
        # backward prediction
        strack_pool_xy = [track.xy_prev for track in tracked_stracks]
        detections_xy_prev = [det.xy_prev for det in detections]

        dists = matching.center_distance(strack_pool_xy, detections_xy_prev)
        
        if self.reid:
            # calculate reid distance
            reid_dists = matching.embedding_distance(tracked_stracks, detections) / 2.0
            # normalize center distance
            dists = np.clip(dists, a_min=0., a_max=self.max_spatial_dist) / self.max_spatial_dist
            
            if self.use_temporal_mixing:
                strack_pool_t = [[track.frame_id] for track in tracked_stracks]
                detections_t = [[self.frame_id-1] for det in detections]
                temp_dist = matching.center_distance(strack_pool_t, detections_t)
                temp_weight = 1 / (self.lambda_1 + np.exp(-1. * self.lambda_2 * temp_dist))
                dists = (1 - temp_weight) * dists + temp_weight * reid_dists
            
            else:
                dists = (1-self.dist_alpha) * dists + reid_dists * self.dist_alpha # 0.5 is the weight for reid distance
            
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.lapjv_thresh)
        
        else:
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.max_spatial_dist)

        for itracked, idet in matches:
            track = tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                if self.reid:
                    track.update(det, self.frame_id, update_feature=True)
                else:
                    track.update(det, self.frame_id, update_feature=False)
                activated_starcks.append(track)
            else:
                if self.reid:
                    track.re_activate(det, self.frame_id, new_id=False, update_feature=True)
                else:
                    track.re_activate(det, self.frame_id, new_id=False, update_feature=False)
                refind_stracks.append(track)
        
        for it in u_track:
            track = tracked_stracks[it]
            missed_stracks.append(track)
        

        '''
            Deal with lost tracks, which are:
            (1) Misdetected tracks
            (2) Long-term lost tracks
            (3) Misdetections in current frame
        '''
        # joint lost stracks as unconfirmed
        re_stracks = joint_stracks(missed_stracks, self.lost_stracks)
        
        u_detections = [detections[i] for i in u_detection]
        u_detections_xy = [det.xy for det in u_detections]
        re_stracks_xy = [track.xy for track in re_stracks]
        
        dists = matching.center_distance(re_stracks_xy, u_detections_xy)
        
        if self.reid:
            # calculate reid distance
            reid_dists = matching.embedding_distance(re_stracks, u_detections) / 2.0
            # normalize center distance
            dists = np.clip(dists, a_min=0., a_max=self.max_spatial_dist2) / self.max_spatial_dist2
            
            if self.use_temporal_mixing:
                strack_pool_t = [[track.frame_id] for track in tracked_stracks]
                detections_t = [[self.frame_id-1] for det in detections]
                temp_dist = matching.center_distance(strack_pool_t, detections_t)
                temp_weight = 1 / (self.lambda_1 + np.exp(-1. * self.lambda_2 * temp_dist))
                dists = (1 - temp_weight) * dists + temp_weight * reid_dists
            
            else:
                dists = (1-self.dist_alpha) * dists + reid_dists * self.dist_alpha # 0.5 is the weight for reid distance
            
            matches, u_lost, u_detection = matching.linear_assignment(dists, thresh=self.lapjv_thresh2)
        
        else:
            matches, u_lost, u_detection = matching.linear_assignment(dists, thresh=self.max_spatial_dist2)

        for itracked, idet in matches:
            track = re_stracks[itracked]
            det = u_detections[idet]
            if track.state == TrackState.Tracked:
                if self.reid:
                    track.update(det, self.frame_id, update_feature=True)
                else:
                    track.update(det, self.frame_id, update_feature=False)
                activated_starcks.append(track)
            else:
                if self.reid:
                    track.re_activate(det, self.frame_id, new_id=False, update_feature=True)
                else:
                    track.re_activate(det, self.frame_id, new_id=False, update_feature=False)
                refind_stracks.append(track)
                
        for it in u_lost:
            track = re_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        detections = [detections[i] for i in u_detection]
        detections_xy = [det.xy_prev for det in detections]
        unconfirmed_xy = [track.xy for track in unconfirmed]
        
        dists = matching.center_distance(unconfirmed_xy, detections_xy)
        
        if self.reid:
            # calculate reid distance
            reid_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            # normalize center distance
            # print(np.max(dists), np.min(dists), np.max(dists)/self.lapjv_thresh2) if len(dists) > 0 and len(dists[0])>0 else None
            dists = np.clip(dists, a_min=0., a_max=self.max_spatial_dist2) / self.max_spatial_dist2
            
            if self.use_temporal_mixing:
                unconfirmed_t = [[track.frame_id] for track in unconfirmed]
                detections_t = [[self.frame_id-1] for det in detections]
                temp_dist = matching.center_distance(unconfirmed_t, detections_t)
                temp_weight = 1 / (self.lambda_1 + np.exp(-1. * self.lambda_2 * temp_dist))
                dists = (1 - temp_weight) * dists + temp_weight * reid_dists
            
            else:
                dists = (1-self.dist_alpha) * dists + reid_dists * self.dist_alpha # 0.5 is the weight for reid distance
            
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.lapjv_thresh2)
        
        else:
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.max_spatial_dist2)
        
        
        for itracked, idet in matches:
            track = unconfirmed[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                if self.reid:
                    track.update(det, self.frame_id, update_feature=True)
                else:
                    track.update(det, self.frame_id, update_feature=False)
                activated_starcks.append(track)
            else:
                if self.reid:
                    track.re_activate(det, self.frame_id, new_id=False, update_feature=True)
                else:
                    track.re_activate(det, self.frame_id, new_id=False, update_feature=False)
                refind_stracks.append(track)
        
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        
        """ Step 3: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            if self.reid:
                track.update_features(reid[inew])
            activated_starcks.append(track)
        
        """ Step 5: Update state"""
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

        # print('===========Frame {}=========='.format(self.frame_id))
        # print('Activated: {}'.format([track.track_id for track in activated_starcks]))
        # print('Refind: {}'.format([track.track_id for track in refind_stracks]))
        # print('Lost: {}'.format([track.track_id for track in lost_stracks]))
        # print('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks


    def update_old(self, dets, dets_prev, score, reid=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        remain_inds = score > self.det_thresh - 0.1
        dets = dets[remain_inds]
        dets_prev = dets_prev[remain_inds]
        
        if self.reid:
            reid = reid[remain_inds]
            assert len(dets) == len(reid), '{} != {}'.format(len(dets), len(reid))

        if len(dets) > 0:
            """Detections"""
            detections = [STrack(xy, xy_prev, s, self.max_time_lost) for
                          (xy, xy_prev, s) in zip(dets, dets_prev, score)]
            
            if self.reid:
                """Update reid features"""
                for track, feat in zip(detections, reid):
                    track.update_features(feat)
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed: List[STrack] = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if track.is_activated:
                tracked_stracks.append(track)
            else:
                unconfirmed.append(track)

        ''' Step 2:association'''
        '''
            Step 2.1: Backward prediction with activated tracks
            Step 2.2: Forward prediction with lost + unmatched tracks
        '''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        strack_pool_xy = [track.xy for track in strack_pool]
        detections_xy_prev = [det.xy for det in detections]

        dists = matching.center_distance(strack_pool_xy, detections_xy_prev)
        
        if self.reid:
            # calculate reid distance
            reid_dists = matching.embedding_distance(strack_pool, detections) / 2.0
            # normalize center distance
            # print(np.max(dists), np.min(dists), np.max(dists) / self.lapjv_thresh) if len(dists) > 0 else None
            dists = np.clip(dists, a_min=0., a_max=self.max_spatial_dist) / self.max_spatial_dist
            
            if self.use_temporal_mixing:
                strack_pool_t = [[track.frame_id] for track in strack_pool]
                detections_t = [[self.frame_id-1] for det in detections]
                temp_dist = matching.center_distance(strack_pool_t, detections_t)
                temp_weight = 1 / (1 + np.exp(self.lambda_1 - 1. * self.lambda_2 * temp_dist))
                dists = (1 - temp_weight) * dists + temp_weight * reid_dists
            
            else:
                dists = (1-self.dist_alpha) * dists + reid_dists * self.dist_alpha # 0.5 is the weight for reid distance
            
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.lapjv_thresh)
        
        else:
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.max_spatial_dist)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                if self.reid:
                    track.update(det, self.frame_id, update_feature=True)
                else:
                    track.update(det, self.frame_id, update_feature=False)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = strack_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        detections_xy = [det.xy_prev for det in detections]
        unconfirmed_xy = [track.xy for track in unconfirmed]
        
        dists = matching.center_distance(unconfirmed_xy, detections_xy)
        
        if self.reid:
            # calculate reid distance
            reid_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            # normalize center distance
            # print(np.max(dists), np.min(dists), np.max(dists)/self.lapjv_thresh2) if len(dists) > 0 and len(dists[0])>0 else None
            dists = np.clip(dists, a_min=0., a_max=self.max_spatial_dist2) / self.max_spatial_dist2
            
            if self.use_temporal_mixing:
                unconfirmed_t = [[track.frame_id] for track in unconfirmed]
                detections_t = [[self.frame_id-1] for det in detections]
                temp_dist = matching.center_distance(unconfirmed_t, detections_t)
                temp_weight = 1 / (1 + np.exp(self.lambda_1 - 1. * self.lambda_2 * temp_dist))
                dists = (1 - temp_weight) * dists + temp_weight * reid_dists
            
            else:
                # dists = dists + reid_dists * 0.5 # 0.5 is the weight for reid distance
                dists = (1-self.dist_alpha) * dists + reid_dists * self.dist_alpha # 0.5 is the weight for reid distance
            
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.lapjv_thresh2)
        
        else:
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.max_spatial_dist2)
        
        
        for itracked, idet in matches:
            if self.reid:
                unconfirmed[itracked].update(detections[idet], self.frame_id, update_feature=True)
            else:
                unconfirmed[itracked].update(detections[idet], self.frame_id, update_feature=False)
            activated_starcks.append(unconfirmed[itracked])
        
        for it in u_unconfirmed:
            track = unconfirmed[it]
            if track.state != TrackState.Lost: # keep lost tracks in the lost list
                track.mark_removed()
                removed_stracks.append(track)

        """ Step 3: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            if self.reid:
                track.update_features(reid[inew])
            activated_starcks.append(track)
        
        """ Step 5: Update state"""
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

        # print('===========Frame {}=========='.format(self.frame_id))
        # print('Activated: {}'.format([track.track_id for track in activated_starcks]))
        # print('Refind: {}'.format([track.track_id for track in refind_stracks]))
        # print('Lost: {}'.format([track.track_id for track in lost_stracks]))
        # print('Removed: {}'.format([track.track_id for track in removed_stracks]))

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
    # track_a = [t.xy_prev for t in stracksa]
    track_a = [t.xy for t in stracksa]
    track_b = [t.xy for t in stracksb]
    pdist = matching.center_distance(track_a, track_b)
    pairs = np.where(pdist < 6)
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
