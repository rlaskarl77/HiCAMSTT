import os
import torch
import numpy as np
import unittest
from unittest.mock import MagicMock, patch
from train_con import WorldTrackModel
import os.path as osp

class TestWorldTrackModel(unittest.TestCase):
    
    def setUp(self):
        self.model = WorldTrackModel(
            learn_cont_pose=False,
        )
        self.model.trainer = MagicMock()
        self.model.trainer.log_dir = 'test_log_dir'
        
    @patch('numpy.loadtxt')
    @patch('os.makedirs')
    @patch('numpy.savetxt')
    @patch.object(WorldTrackModel, 'logger', new_callable=MagicMock)
    def test_on_test_epoch_end_tracking(self, mock_logger, mock_savetxt, mock_makedirs, mock_loadtxt):
        self.model.test_mode = 'tracking'
        self.model.test_dataset = 'wildtrack'
        self.model.moda_pred_list = np.random.randint(0, 10, (10, 10))
        self.model.moda_gt_list = np.random.randint(0, 10, (10, 10))
        self.model.mota_pred_list = np.random.randint(0, 10, (10, 11))
        self.model.mota_gt_list = np.random.randint(0, 10, (10, 11))
        
        self.model.on_test_epoch_end()
        
        mock_savetxt.assert_any_call(
            osp.join('test_log_dir', 'moda_pred.txt'), 
            np.array(self.model.moda_pred_list), '%f', delimiter=' ', newline='\n'
        )
        mock_savetxt.assert_any_call(
            osp.join('test_log_dir', 'moda_gt.txt'), 
            np.array(self.model.moda_gt_list), '%d', delimiter=' ', newline='\n'
        )
        mock_savetxt.assert_any_call(
            osp.join('test_log_dir', 'mota_pred.txt'), 
            np.array(self.model.mota_pred_list), '%f', delimiter=','
        )
        mock_savetxt.assert_any_call(
            osp.join('test_log_dir', 'mota_gt.txt'), 
            np.array(self.model.mota_gt_list), '%f', delimiter=','
        )
        
        mock_loadtxt.assert_any_call(osp.join('test_log_dir', 'moda_pred.txt'), delimiter=' ')
        mock_loadtxt.assert_any_call(osp.join('test_log_dir', 'moda_gt.txt'), delimiter=' ')
        
    @patch('os.makedirs')
    @patch('numpy.savetxt')
    @patch.object(WorldTrackModel, 'logger', new_callable=MagicMock)
    def test_on_test_epoch_end_feature_distance(self, mock_logger, mock_savetxt, mock_makedirs):
        self.model.test_mode = 'feature_distance'
        self.model.feat_correct_self = 10
        self.model.feat_total_self = 20
        self.model.feat_correct_random = 5
        self.model.feat_total_random = 15
        self.model.feat_dist_self = torch.full((100, 100), -1, dtype=torch.float32)
        self.model.feat_dist_random = torch.full((100, 100), -1, dtype=torch.float32)
        self.model.feat_dist_count_self = torch.zeros(100, 100)
        self.model.feat_dist_count_random = torch.zeros(100, 100)
        
        self.model.on_test_epoch_end()
        
        mock_logger.experiment.add_figure.assert_called()
        
    @patch('os.makedirs')
    @patch('numpy.load')
    @patch('os.listdir')
    @patch.object(WorldTrackModel, 'logger', new_callable=MagicMock)
    def test_on_test_epoch_end_save_features(self, mock_logger, mock_listdir, mock_load, mock_makedirs):
        self.model.test_mode = 'save_features'
        mock_listdir.return_value = ['1_1.npy', '2_2.npy', '3_3.npy', '4_4.npy', '5_5.npy', '6_6.npy']
        mock_load.side_effect = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9]), np.array([10, 11, 12]), np.array([13, 14, 15]), np.array([16, 17, 18])]
        
        self.model.on_test_epoch_end()
        
        mock_logger.experiment.add_figure.assert_called()

    @patch('os.makedirs')
    @patch('numpy.load')
    @patch('os.listdir')
    @patch.object(WorldTrackModel, 'logger', new_callable=MagicMock)
    def test_on_test_epoch_end_pose(self, mock_logger, mock_listdir, mock_load, mock_makedirs):
        self.model.test_mode = 'pose'
        mock_listdir.return_value = ['pose_1_1_1.npy', 'pose_2_2_2.npy']
        mock_load.side_effect = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        
        self.model.on_test_epoch_end()
        
        mock_logger.experiment.add_figure.assert_called()

if __name__ == '__main__':
    unittest.main()
