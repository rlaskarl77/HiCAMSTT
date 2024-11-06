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
