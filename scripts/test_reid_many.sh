/home/TrackTacular/experiments/lightning_logs/train/multiviewx/mvdet/baseline/checkpoints/model-epoch=29-val_loss=9.58-val_center=2.16.ckpt

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_multiviewx_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_baseline.yml \
    -c configs/hyp/h_test_tracking_noreid.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/multiviewx/mvdet/baseline_tracking' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/multiviewx/mvdet/baseline/checkpoints/model-epoch=29-val_loss=9.58-val_center=2.16.ckpt' \
    > logs/cvpr2025/test_multiviewx_mvdet_baseline_tracking.log 2>&1

/home/TrackTacular/experiments/lightning_logs/train/multiviewx/mvdet/simclr@t0.03_pseudolabel_hardmask/checkpoints/model-epoch=29-val_loss=14.14-val_center=2.33.ckpt

CUDA_VISIBLE_DEVICES=1 python -u train_con.py test \
    -c configs/d_multiviewx_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_hardmask.yml \
    -c configs/hyp/h_test_tracking_noreid.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/multiviewx/mvdet/simclr@t0.03_pseudolabel_hardmask_tracking_noreid' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/multiviewx/mvdet/simclr@t0.03_pseudolabel_hardmask/checkpoints/model-epoch=29-val_loss=14.14-val_center=2.33.ckpt' \
    > logs/cvpr2025/test_multiviewx_mvdet_simclr@t0.03_pseudolabel_hardmask_tracking_noreid.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u train_con.py test \
    -c configs/d_multiviewx_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_hardmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.25_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/multiviewx/mvdet/simclr@t0.03_pseudolabel_hardmask_tracking_notemp@a0.5_lapjv@0.25_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/multiviewx/mvdet/simclr@t0.03_pseudolabel_hardmask/checkpoints/model-epoch=29-val_loss=14.14-val_center=2.33.ckpt' \
    > logs/cvpr2025/test_multiviewx_mvdet_simclr@t0.03_pseudolabel_hardmask_tracking_notemp@a0.5_lapjv@0.25_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u train_con.py test \
    -c configs/d_multiviewx_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_hardmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.5_0.75.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/multiviewx/mvdet/simclr@t0.03_pseudolabel_hardmask_tracking_notemp@a0.5_lapjv@0.5_0.75' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/multiviewx/mvdet/simclr@t0.03_pseudolabel_hardmask/checkpoints/model-epoch=29-val_loss=14.14-val_center=2.33.ckpt' \
    > logs/cvpr2025/test_multiviewx_mvdet_simclr@t0.03_pseudolabel_hardmask_tracking_notemp@a0.5_lapjv@0.5_0.75.log 2>&1

/home/TrackTacular/experiments/lightning_logs/train/multiviewx/mvdet/simclr@t0.03_pseudolabel_nomask/checkpoints/model-epoch=29-val_loss=13.34-val_center=2.65.ckpt

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_multiviewx_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_nomask.yml \
    -c configs/hyp/h_test_tracking_noreid.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/multiviewx/mvdet/simclr@t0.03_pseudolabel_nomask_tracking_noreid' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/multiviewx/mvdet/simclr@t0.03_pseudolabel_nomask/checkpoints/model-epoch=29-val_loss=13.34-val_center=2.65.ckpt' \
    > logs/cvpr2025/test_multiviewx_mvdet_simclr@t0.03_pseudolabel_nomask_tracking_noreid.log 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_multiviewx_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_nomask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.25_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/multiviewx/mvdet/simclr@t0.03_pseudolabel_nomask_tracking_notemp@a0.5_lapjv@0.25_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/multiviewx/mvdet/simclr@t0.03_pseudolabel_nomask/checkpoints/model-epoch=29-val_loss=13.34-val_center=2.65.ckpt' \
    > logs/cvpr2025/test_multiviewx_mvdet_simclr@t0.03_pseudolabel_nomask_tracking_notemp@a0.5_lapjv@0.25_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_multiviewx_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_nomask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.5_0.75.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/multiviewx/mvdet/simclr@t0.03_pseudolabel_nomask_tracking_notemp@a0.5_lapjv@0.5_0.75' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/multiviewx/mvdet/simclr@t0.03_pseudolabel_nomask/checkpoints/model-epoch=29-val_loss=13.34-val_center=2.65.ckpt' \
    > logs/cvpr2025/test_multiviewx_mvdet_simclr@t0.03_pseudolabel_nomask_tracking_notemp@a0.5_lapjv@0.5_0.75.log 2>&1

/home/TrackTacular/experiments/lightning_logs/train/multiviewx/mvdet/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=12.23-val_center=2.35.ckpt

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_multiviewx_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_noreid.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/multiviewx/mvdet/simclr@t0.03_pseudolabel_softmask_tracking_noreid' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/multiviewx/mvdet/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=12.23-val_center=2.35.ckpt' \
    > logs/cvpr2025/test_multiviewx_mvdet_simclr@t0.03_pseudolabel_softmask_tracking_noreid.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_multiviewx_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.25_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/multiviewx/mvdet/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.25_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/multiviewx/mvdet/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=12.23-val_center=2.35.ckpt' \
    > logs/cvpr2025/test_multiviewx_mvdet_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.25_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_multiviewx_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.5_0.75.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/multiviewx/mvdet/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.5_0.75' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/multiviewx/mvdet/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=12.23-val_center=2.35.ckpt' \
    > logs/cvpr2025/test_multiviewx_mvdet_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.5_0.75.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_multiviewx_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.2_0.33.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/multiviewx/mvdet/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.2_0.33' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/multiviewx/mvdet/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=12.23-val_center=2.35.ckpt' \
    > logs/cvpr2025/test_multiviewx_mvdet_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.2_0.33.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_multiviewx_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.33_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/multiviewx/mvdet/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.33_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/multiviewx/mvdet/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=12.23-val_center=2.35.ckpt' \
    > logs/cvpr2025/test_multiviewx_mvdet_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.33_0.5.log 2>&1

/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_onlyid/checkpoints/model-epoch=29-val_loss=12.43-val_center=5.24.ckpt

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_onlyid.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.33_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_onlyid_tracking_notemp@a0.5_lapjv@0.33_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_onlyid/checkpoints/model-epoch=29-val_loss=12.43-val_center=5.24.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_onlyid_tracking_notemp@a0.5_lapjv@0.33_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_onlyid.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.25_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_onlyid_tracking_notemp@a0.5_lapjv@0.25_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_onlyid/checkpoints/model-epoch=29-val_loss=12.43-val_center=5.24.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_onlyid_tracking_notemp@a0.5_lapjv@0.25_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_onlyid.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.5_0.75.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_onlyid_tracking_notemp@a0.5_lapjv@0.5_0.75' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_onlyid/checkpoints/model-epoch=29-val_loss=12.43-val_center=5.24.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_onlyid_tracking_notemp@a0.5_lapjv@0.5_0.75.log 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_onlyid.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.2_0.33.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_onlyid_tracking_notemp@a0.5_lapjv@0.2_0.33' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_onlyid/checkpoints/model-epoch=29-val_loss=12.43-val_center=5.24.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_onlyid_tracking_notemp@a0.5_lapjv@0.2_0.33.log 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_onlyid.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.67_0.8.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_onlyid_tracking_notemp@a0.5_lapjv@0.67_0.8' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_onlyid/checkpoints/model-epoch=29-val_loss=12.43-val_center=5.24.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_onlyid_tracking_notemp@a0.5_lapjv@0.67_0.8.log 2>&1

/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_hardmask/checkpoints/model-epoch=29-val_loss=14.24-val_center=5.27.ckpt

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_hardmask.yml \
    -c configs/hyp/h_test_tracking_noreid.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_hardmask_tracking_noreid' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_hardmask/checkpoints/model-epoch=29-val_loss=14.24-val_center=5.27.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_hardmask_tracking_noreid.log 2>&1

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_hardmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.1_lapjv@0.25_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_hardmask_tracking_notemp@a0.1_lapjv@0.25_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_hardmask/checkpoints/model-epoch=29-val_loss=14.24-val_center=5.27.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_hardmask_tracking_notemp@a0.1_lapjv@0.25_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_hardmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.25_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_hardmask_tracking_notemp@a0.5_lapjv@0.25_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_hardmask/checkpoints/model-epoch=29-val_loss=14.24-val_center=5.27.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_hardmask_tracking_notemp@a0.5_lapjv@0.25_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_hardmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.9_lapjv@0.25_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_hardmask_tracking_notemp@a0.9_lapjv@0.25_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_hardmask/checkpoints/model-epoch=29-val_loss=14.24-val_center=5.27.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_hardmask_tracking_notemp@a0.9_lapjv@0.25_0.5.log 2>&1

# 

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_hardmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.2_0.33.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_hardmask_tracking_notemp@a0.5_lapjv@0.2_0.33' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_hardmask/checkpoints/model-epoch=29-val_loss=14.24-val_center=5.27.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_hardmask_tracking_notemp@a0.5_lapjv@0.2_0.33.log 2>&1

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_hardmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.5_0.75.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_hardmask_tracking_notemp@a0.5_lapjv@0.5_0.75' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_hardmask/checkpoints/model-epoch=29-val_loss=14.24-val_center=5.27.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_hardmask_tracking_notemp@a0.5_lapjv@0.5_0.75.log 2>&1

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_hardmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.33_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_hardmask_tracking_notemp@a0.5_lapjv@0.33_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_hardmask/checkpoints/model-epoch=29-val_loss=14.24-val_center=5.27.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_hardmask_tracking_notemp@a0.5_lapjv@0.33_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_hardmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.75_1.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_hardmask_tracking_notemp@a0.5_lapjv@0.75_1' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_hardmask/checkpoints/model-epoch=29-val_loss=14.24-val_center=5.27.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_hardmask_tracking_notemp@a0.5_lapjv@0.75_1.log 2>&1

/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_nomask/checkpoints/model-epoch=29-val_loss=14.35-val_center=5.17.ckpt

CUDA_VISIBLE_DEVICES=1 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_nomask.yml \
    -c configs/hyp/h_test_tracking_noreid.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_nomask_tracking_noreid' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_nomask/checkpoints/model-epoch=29-val_loss=14.35-val_center=5.17.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_nomask_tracking_noreid.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_nomask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.1_lapjv@0.25_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_nomask_tracking_notemp@a0.1_lapjv@0.25_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_nomask/checkpoints/model-epoch=29-val_loss=14.35-val_center=5.17.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_nomask_tracking_notemp@a0.1_lapjv@0.25_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_nomask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.25_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_nomask_tracking_notemp@a0.5_lapjv@0.25_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_nomask/checkpoints/model-epoch=29-val_loss=14.35-val_center=5.17.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_nomask_tracking_notemp@a0.5_lapjv@0.25_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_nomask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.9_lapjv@0.25_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_nomask_tracking_notemp@a0.9_lapjv@0.25_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_nomask/checkpoints/model-epoch=29-val_loss=14.35-val_center=5.17.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_nomask_tracking_notemp@a0.9_lapjv@0.25_0.5.log 2>&1

#

CUDA_VISIBLE_DEVICES=1 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_nomask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.2_0.33.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_nomask_tracking_notemp@a0.5_lapjv@0.2_0.33' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_nomask/checkpoints/model-epoch=29-val_loss=14.35-val_center=5.17.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_nomask_tracking_notemp@a0.5_lapjv@0.2_0.33.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_nomask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.5_0.75.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_nomask_tracking_notemp@a0.5_lapjv@0.5_0.75' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_nomask/checkpoints/model-epoch=29-val_loss=14.35-val_center=5.17.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_nomask_tracking_notemp@a0.5_lapjv@0.5_0.75.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_nomask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.33_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_nomask_tracking_notemp@a0.5_lapjv@0.33_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_nomask/checkpoints/model-epoch=29-val_loss=14.35-val_center=5.17.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_nomask_tracking_notemp@a0.5_lapjv@0.33_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_nomask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.75_1.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_nomask_tracking_notemp@a0.5_lapjv@0.75_1' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_nomask/checkpoints/model-epoch=29-val_loss=14.35-val_center=5.17.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_nomask_tracking_notemp@a0.5_lapjv@0.75_1.log 2>&1

/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.30-val_center=5.31.ckpt

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_noreid.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_tracking_noreid' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.30-val_center=5.31.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_tracking_noreid.log 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.1_lapjv@0.25_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.1_lapjv@0.25_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.30-val_center=5.31.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.1_lapjv@0.25_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.25_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.25_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.30-val_center=5.31.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.25_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.9_lapjv@0.25_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.9_lapjv@0.25_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.30-val_center=5.31.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.9_lapjv@0.25_0.5.log 2>&1

#

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.2_0.33.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.2_0.33' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.30-val_center=5.31.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.2_0.33.log 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.5_0.75.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.5_0.75' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.30-val_center=5.31.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.5_0.75.log 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.33_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.33_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.30-val_center=5.31.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.33_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.75_1.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.75_1' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.30-val_center=5.31.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.75_1.log 2>&1


############################################################################################################
/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_nomask/checkpoints/model-epoch=29-val_loss=15.73-val_center=5.04.ckpt


CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_cont@t0.03th0.8_nomask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.9_lapjv@0.25_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_nomask_tracking_notemp@a0.9_lapjv@0.25_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_nomask/checkpoints/model-epoch=29-val_loss=15.73-val_center=5.04.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_cont@t0.03th0.8_nomask_tracking_notemp@a0.9_lapjv@0.25_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_cont@t0.03th0.8_nomask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.5_0.75.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_nomask_tracking_notemp@a0.5_lapjv@0.5_0.75' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_nomask/checkpoints/model-epoch=29-val_loss=15.73-val_center=5.04.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_cont@t0.03th0.8_nomask_tracking_notemp@a0.5_lapjv@0.5_0.75.log 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_cont@t0.03th0.8_nomask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.2_0.33.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_nomask_tracking_notemp@a0.5_lapjv@0.2_0.33' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_nomask/checkpoints/model-epoch=29-val_loss=15.73-val_center=5.04.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_cont@t0.03th0.8_nomask_tracking_notemp@a0.5_lapjv@0.2_0.33.log 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_cont@t0.03th0.8_nomask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.33_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_nomask_tracking_notemp@a0.5_lapjv@0.33_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_nomask/checkpoints/model-epoch=29-val_loss=15.73-val_center=5.04.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_cont@t0.03th0.8_nomask_tracking_notemp@a0.5_lapjv@0.33_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_cont@t0.03th0.8_nomask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.75_1.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_nomask_tracking_notemp@a0.5_lapjv@0.75_1' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_nomask/checkpoints/model-epoch=29-val_loss=15.73-val_center=5.04.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_cont@t0.03th0.8_nomask_tracking_notemp@a0.5_lapjv@0.75_1.log 2>&1


############################################################################################################
/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_softmask/checkpoints/model-epoch=29-val_loss=15.68-val_center=5.19.ckpt


CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_cont@t0.03th0.8_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.9_lapjv@0.25_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_softmask_tracking_notemp@a0.9_lapjv@0.25_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_softmask/checkpoints/model-epoch=29-val_loss=15.68-val_center=5.19.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_cont@t0.03th0.8_softmask_tracking_notemp@a0.9_lapjv@0.25_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_cont@t0.03th0.8_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.5_0.75.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_softmask_tracking_notemp@a0.5_lapjv@0.5_0.75' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_softmask/checkpoints/model-epoch=29-val_loss=15.68-val_center=5.19.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_cont@t0.03th0.8_softmask_tracking_notemp@a0.5_lapjv@0.5_0.75.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_cont@t0.03th0.8_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.2_0.33.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_softmask_tracking_notemp@a0.5_lapjv@0.2_0.33' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_softmask/checkpoints/model-epoch=29-val_loss=15.68-val_center=5.19.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_cont@t0.03th0.8_softmask_tracking_notemp@a0.5_lapjv@0.2_0.33.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_cont@t0.03th0.8_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.33_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_softmask_tracking_notemp@a0.5_lapjv@0.33_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_softmask/checkpoints/model-epoch=29-val_loss=15.68-val_center=5.19.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_cont@t0.03th0.8_softmask_tracking_notemp@a0.5_lapjv@0.33_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_cont@t0.03th0.8_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.75_1.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_softmask_tracking_notemp@a0.5_lapjv@0.75_1' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_softmask/checkpoints/model-epoch=29-val_loss=15.68-val_center=5.19.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_cont@t0.03th0.8_softmask_tracking_notemp@a0.5_lapjv@0.75_1.log 2>&1


############################################################################################################
/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_hardmask/checkpoints/model-epoch=29-val_loss=15.92-val_center=5.18.ckpt


CUDA_VISIBLE_DEVICES=1 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_cont@t0.03th0.8_hardmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.9_lapjv@0.25_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_hardmask_tracking_notemp@a0.9_lapjv@0.25_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_hardmask/checkpoints/model-epoch=29-val_loss=15.92-val_center=5.18.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_cont@t0.03th0.8_hardmask_tracking_notemp@a0.9_lapjv@0.25_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_cont@t0.03th0.8_hardmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.5_0.75.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_hardmask_tracking_notemp@a0.5_lapjv@0.5_0.75' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_hardmask/checkpoints/model-epoch=29-val_loss=15.92-val_center=5.18.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_cont@t0.03th0.8_hardmask_tracking_notemp@a0.5_lapjv@0.5_0.75.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_cont@t0.03th0.8_hardmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.2_0.33.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_hardmask_tracking_notemp@a0.5_lapjv@0.2_0.33' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_hardmask/checkpoints/model-epoch=29-val_loss=15.92-val_center=5.18.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_cont@t0.03th0.8_hardmask_tracking_notemp@a0.5_lapjv@0.2_0.33.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_cont@t0.03th0.8_hardmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.33_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_hardmask_tracking_notemp@a0.5_lapjv@0.33_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_hardmask/checkpoints/model-epoch=29-val_loss=15.92-val_center=5.18.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_cont@t0.03th0.8_hardmask_tracking_notemp@a0.5_lapjv@0.33_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_cont@t0.03th0.8_hardmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.75_1.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_hardmask_tracking_notemp@a0.5_lapjv@0.75_1' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_cont@t0.03th0.8_hardmask/checkpoints/model-epoch=29-val_loss=15.92-val_center=5.18.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_cont@t0.03th0.8_hardmask_tracking_notemp@a0.5_lapjv@0.75_1.log 2>&1


/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache/checkpoints/model-epoch=29-val_loss=14.58-val_center=5.18.ckpt

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_noreid.yml \
    -c configs/hyp/h_nocache.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_nomask_tracking_noreid' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache/checkpoints/model-epoch=29-val_loss=14.58-val_center=5.18.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_nocache_tracking_noreid.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.5_0.75.yml \
    -c configs/hyp/h_nocache.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache_tracking_notemp@a0.5_lapjv@0.5_0.75' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache/checkpoints/model-epoch=29-val_loss=14.58-val_center=5.18.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_nocache_tracking_notemp@a0.5_lapjv@0.5_0.75.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.9_lapjv@0.25_0.5.yml \
    -c configs/hyp/h_nocache.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache_tracking_notemp@a0.9_lapjv@0.25_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache/checkpoints/model-epoch=29-val_loss=14.58-val_center=5.18.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_nocache_tracking_notemp@a0.9_lapjv@0.25_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.2_0.33.yml \
    -c configs/hyp/h_nocache.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache_tracking_notemp@a0.5_lapjv@0.2_0.33' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache/checkpoints/model-epoch=29-val_loss=14.58-val_center=5.18.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_nocache_tracking_notemp@a0.5_lapjv@0.2_0.33.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.33_0.5.yml \
    -c configs/hyp/h_nocache.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache_tracking_notemp@a0.5_lapjv@0.33_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache/checkpoints/model-epoch=29-val_loss=14.58-val_center=5.18.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_nocache_tracking_notemp@a0.5_lapjv@0.33_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.75_1.yml \
    -c configs/hyp/h_nocache.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache_tracking_notemp@a0.5_lapjv@0.75_1' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache/checkpoints/model-epoch=29-val_loss=14.58-val_center=5.18.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_nocache_tracking_notemp@a0.5_lapjv@0.75_1.log 2>&1



/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache_120e_batch8/checkpoints/model-epoch=77-val_loss=14.84-val_center=4.99.ckpt

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_noreid.yml \
    -c configs/hyp/h_nocache.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache_120e_batch8_tracking_noreid' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache_120e_batch8/checkpoints/model-epoch=77-val_loss=14.84-val_center=4.99.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_nocache_120e_batch8_tracking_noreid.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.5_0.75.yml \
    -c configs/hyp/h_nocache.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache_120e_batch8_tracking_notemp@a0.5_lapjv@0.5_0.75' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache_120e_batch8/checkpoints/model-epoch=77-val_loss=14.84-val_center=4.99.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_nocache_120e_batch8_tracking_notemp@a0.5_lapjv@0.5_0.75.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.25_0.5.yml \
    -c configs/hyp/h_nocache.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache_120e_batch8_tracking_notemp@a0.5_lapjv@0.25_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache_120e_batch8/checkpoints/model-epoch=77-val_loss=14.84-val_center=4.99.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_nocache_120e_batch8_tracking_notemp@a0.5_lapjv@0.25_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.2_0.33.yml \
    -c configs/hyp/h_nocache.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache_120e_batch8_tracking_notemp@a0.5_lapjv@0.2_0.33' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache_120e_batch8/checkpoints/model-epoch=77-val_loss=14.84-val_center=4.99.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_nocache_120e_batch8_tracking_notemp@a0.5_lapjv@0.2_0.33.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.33_0.5.yml \
    -c configs/hyp/h_nocache.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache_120e_batch8_tracking_notemp@a0.5_lapjv@0.33_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache_120e_batch8/checkpoints/model-epoch=77-val_loss=14.84-val_center=4.99.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_nocache_120e_batch8_tracking_notemp@a0.5_lapjv@0.33_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.75_1.yml \
    -c configs/hyp/h_nocache.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache_120e_batch8_tracking_notemp@a0.5_lapjv@0.75_1' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet/simclr@t0.03_pseudolabel_softmask_nocache_120e_batch8/checkpoints/model-epoch=77-val_loss=14.84-val_center=4.99.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_simclr@t0.03_pseudolabel_softmask_nocache_120e_batch8_tracking_notemp@a0.5_lapjv@0.75_1.log 2>&1







/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet_res34/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.58-val_center=5.06.ckpt

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet_res34.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.5_0.75.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet_res34/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.5_0.75' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet_res34/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.58-val_center=5.06.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_res34_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.5_0.75.log 2>&1

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet_res34.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_noreid.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet_res34/simclr@t0.03_pseudolabel_softmask_tracking_noreid' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet_res34/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.58-val_center=5.06.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_res34_simclr@t0.03_pseudolabel_softmask_tracking_noreid.log 2>&1

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet_res34.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.25_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet_res34/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.25_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet_res34/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.58-val_center=5.06.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_res34_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.25_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet_res34.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.2_0.33.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet_res34/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.2_0.33' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet_res34/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.58-val_center=5.06.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_res34_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.2_0.33.log 2>&1

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet_res34.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.33_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet_res34/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.33_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet_res34/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.58-val_center=5.06.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_res34_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.33_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet_res34.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.75_1.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet_res34/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.75_1' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet_res34/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.58-val_center=5.06.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_res34_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.75_1.log 2>&1






/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet_res50/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.89-val_center=5.06.ckpt

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet_res50.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.5_0.75.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet_res50/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.5_0.75' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet_res50/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.89-val_center=5.06.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_res50_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.5_0.75.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet_res50.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_noreid.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet_res50/simclr@t0.03_pseudolabel_softmask_tracking_noreid' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet_res50/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.89-val_center=5.06.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_res50_simclr@t0.03_pseudolabel_softmask_tracking_noreid.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet_res50.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.25_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet_res50/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.25_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet_res50/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.89-val_center=5.06.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_res50_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.25_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet_res50.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.2_0.33.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet_res50/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.2_0.33' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet_res50/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.89-val_center=5.06.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_res50_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.2_0.33.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet_res50.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.33_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet_res50/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.33_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet_res50/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.89-val_center=5.06.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_res50_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.33_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet_res50.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.75_1.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/mvdet_res50/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.75_1' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/mvdet_res50/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=29-val_loss=14.89-val_center=5.06.ckpt' \
    > logs/cvpr2025/test_wildtrack_mvdet_res50_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.75_1.log 2>&1





/home/TrackTacular/experiments/lightning_logs/train/wildtrack/bevformer/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=24-val_loss=18.83-val_center=6.93.ckpt

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_bevformer.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.5_0.75.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/bevformer/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.5_0.75' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/bevformer/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=24-val_loss=18.83-val_center=6.93.ckpt' \
    > logs/cvpr2025/test_wildtrack_bevformer_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.5_0.75.log 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_bevformer.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_noreid.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/bevformer/simclr@t0.03_pseudolabel_softmask_tracking_noreid' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/bevformer/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=24-val_loss=18.83-val_center=6.93.ckpt' \
    > logs/cvpr2025/test_wildtrack_bevformer_simclr@t0.03_pseudolabel_softmask_tracking_noreid.log 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_bevformer.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.25_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/bevformer/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.25_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/bevformer/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=24-val_loss=18.83-val_center=6.93.ckpt' \
    > logs/cvpr2025/test_wildtrack_bevformer_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.25_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_bevformer.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.2_0.33.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/bevformer/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.2_0.33' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/bevformer/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=24-val_loss=18.83-val_center=6.93.ckpt' \
    > logs/cvpr2025/test_wildtrack_bevformer_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.2_0.33.log 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_bevformer.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.33_0.5.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/bevformer/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.33_0.5' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/bevformer/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=24-val_loss=18.83-val_center=6.93.ckpt' \
    > logs/cvpr2025/test_wildtrack_bevformer_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.33_0.5.log 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_con.py test \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_bevformer.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.75_1.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/wildtrack/bevformer/simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.75_1' \
    --ckpt '/home/TrackTacular/experiments/lightning_logs/train/wildtrack/bevformer/simclr@t0.03_pseudolabel_softmask/checkpoints/model-epoch=24-val_loss=18.83-val_center=6.93.ckpt' \
    > logs/cvpr2025/test_wildtrack_bevformer_simclr@t0.03_pseudolabel_softmask_tracking_notemp@a0.5_lapjv@0.75_1.log 2>&1