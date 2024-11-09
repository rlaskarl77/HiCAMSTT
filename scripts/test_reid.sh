CUDA_VISIBLE_DEVICES=0 python -u test_con.py test \
    -c configs/d_hdc.yml \
    -c configs/m_mvdet.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir 'test_tracker' \
    --trainer.logger.version 'mvdet_res18_hdc_simclr_t1_tracking' \
    --model.temperature 1.0 \
    --ckpt reid/train/lightning_logs/mvdet_res18_hdc_simclr_t1/checkpoints/model-epoch=61-val_loss=19.19-val_center=9.78.ckpt \
    > logs/test_tracker.log 2>&1

CUDA_VISIBLE_DEVICES=0 python -u test_con.py test \
    -c configs/d_hdc.yml \
    -c configs/m_mvdet.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir 'test_tracker' \
    --trainer.logger.version '0-50' \
    --model.temperature 1.0 \
    --ckpt reid/train/lightning_logs/mvdet_res18_hdc_simclr_t1/checkpoints/model-epoch=61-val_loss=19.19-val_center=9.78.ckpt \
    > logs/test_tracker_0-50.log 2>&1

CUDA_VISIBLE_DEVICES=0 python -u test_con.py test \
    -c configs/d_hdc.yml \
    -c configs/m_mvdet.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir 'test_tracker' \
    --trainer.logger.version '0-50-C-TMv2' \
    --model.temperature 1.0 \
    --ckpt reid/train/lightning_logs/mvdet_res18_hdc_simclr_t1/checkpoints/model-epoch=61-val_loss=19.19-val_center=9.78.ckpt \
    > logs/test_tracker_0-50-C-TMv2.log 2>&1

python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_cont@t0.03th0.8_softmask.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_softmask' \
    --ckpt '/131_data/namgi/logs/MCMOT/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_softmask/checkpoints/model-epoch=29-val_loss=15.68-val_center=5.19.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_cont@t0.03th0.8_softmask_tracking.log 2>&1

python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_cont@t0.03th0.8_hardmask.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_hardmask' \
    --ckpt '/131_data/namgi/logs/MCMOT/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_hardmask/checkpoints/model-epoch=29-val_loss=15.92-val_center=5.18.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_cont@t0.03th0.8_hardmask_tracking.log 2>&1

python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_cont@t0.03th0.8_hardmask.yml \
    -c configs/hyp/h_test_tracking.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_hardmask_base' \
    --ckpt '/131_data/namgi/logs/MCMOT/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_hardmask/checkpoints/model-epoch=29-val_loss=15.92-val_center=5.18.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_cont@t0.03th0.8_hardmask_tracking_base.log 2>&1

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_cont@t0.03th0.8_hardmask.yml \
    -c configs/hyp/h_test_save_features.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_hardmask_tsne' \
    --ckpt '/131_data/namgi/logs/MCMOT/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_hardmask/checkpoints/model-epoch=29-val_loss=15.92-val_center=5.18.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_cont@t0.03th0.8_hardmask_tsne.log 2>&1


CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_baseline.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/baseline_tracking' \
    --ckpt '/131_data/namgi/logs/MCMOT/lightning_logs/train/wildtrack/mvdet/baseline/checkpoints/model-epoch=29-val_loss=10.49-val_center=5.28.ckpt' \
    > logs/cvpr2025/test_wildtrack_baseline.log 2>&1

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test     -c configs/d_wildtrack_server.yml     -c confi
gs/m_mvdet.yml     -c configs/hyp/h_baseline.yml     -c configs/hyp/h_test_save_features.yml     --trainer.logger TensorBoardLogger     --trainer.logg
er.save_dir '/131_data/namgi/logs/MCMOT'     --trainer.logger.version 'test/wildtrack/mvdet/baseline_tracking'     --ckpt '/131_data/namgi/logs/MCMOT/
lightning_logs/train/wildtrack/mvdet/baseline/checkpoints/model-epoch=29-val_loss=10.49-val_center=5.28.ckpt'     > logs/cvpr2025/test_wildtrack_basel
ine.log 2>&1

/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/baseline/checkpoints/model-epoch=29-val_loss=10.49-val_center=5.28.ckpt

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_baseline.yml \
    -c configs/hyp/h_test_save_features.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/baseline_tsne' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/baseline/checkpoints/model-epoch=29-val_loss=10.49-val_center=5.28.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_baseline_tsne.log 2>&1

/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_80e/checkpoints/model-epoch=47-val_loss=12.90-val_center=5.12.ckpt

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_wildtrack_server_80e.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_save_features.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_80e_tsne' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_80e/checkpoints/model-epoch=47-val_loss=12.90-val_center=5.12.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_80e_tsne.log 2>&1

/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_80e_batch8/checkpoints/model-epoch=70-val_loss=15.19-val_center=5.08.ckpt

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_wildtrack_server_80e.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_save_features.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_80e_batch8_tsne' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_80e_batch8/checkpoints/model-epoch=70-val_loss=15.19-val_center=5.08.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_80e_batch8_tsne.log 2>&1

/home/TrackTacular/experiments/lightning_logs/train/multiviewx/mvdet/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=12.23-val_center=2.35.ckpt

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_multiviewx_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_save_features.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/multiviewx/mvdet/simclr@t0.03_pseudolabel_softmask_tsne' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/multiviewx/mvdet/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=12.23-val_center=2.35.ckpt' \
    > logs/cvpr2025/test_multiviewx_mvdet_simclr@t0.03_pseudolabel_softmask_tsne.log 2>&1

/home/TrackTacular/experiments/lightning_logs/train/multiviewx/mvdet/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=12.23-val_center=2.35.ckpt

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_multiviewx_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_save_features.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/multiviewx/mvdet/simclr@t0.03_pseudolabel_softmask_120e_batch8_tsne' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/multiviewx/mvdet/simclr@t0.03_pseudolabel_softmask_120e_batch8/checkpoints/model-epoch=113-val_loss=10.40-val_center=1.90.ckpt' \
    > logs/cvpr2025/test_multiviewx_mvdet_simclr@t0.03_pseudolabel_softmask_120e_batch8_tsne.log 2>&1

/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_120e_batch4x8/checkpoints/model-epoch=98-val_loss=15.63-val_center=4.96.

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_save_features.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/multiviewx/mvdet/simclr@t0.03_pseudolabel_softmask_120e_batch8_tsne' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_120e_batch4x8/checkpoints/model-epoch=98-val_loss=15.63-val_center=4.96.' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_120e_batch4x8_tsne.log 2>&1