

cd WorldTrack
python -u train_con.py fit\
    -c configs/t_fit_multi_gpu.yml \
    -c configs/d_factory.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_baseline.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/data/namgi/logs/MCMOT' \
    --trainer.logger.version 'train/factory/mvdet/baseline' \
    > logs/train_factory_mvdet_baseline.log 2>&1


CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_factory.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_baseline.yml \
    -c configs/hyp/h_test_tracking_noreid.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/factory/mvdet/baseline' \
    --ckpt '/home/namgi/HiCAMSTT/exp/lightning_logs/train/factory/mvdet/baseline/checkpoints/model-epoch=51-val_loss=7.70-val_center=4.07.ckpt' \
    > logs/test_factory_mvdet_baseline_noreid.log 2>&1


############################################################################################################
# reid

cd WorldTrack
python -u train_con.py fit\
    -c configs/t_fit_multi_gpu.yml \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/data/namgi/logs/MCMOT' \
    --trainer.logger.version 'train/factory/mvdet/simclr@t0.03' \
    > logs/train_factory_mvdet_simclr@t0.03.log 2>&1


CUDA_VISIBLE_DEVICES=0 python -u train_con.py test \
    -c configs/d_factory.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_noreid.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/factory/mvdet/noreid' \
    --ckpt '/131_data/namgi/logs/MCMOT/lightning_logs/train/factory07/mvdet/simclr@t0.03_60e/checkpoints/model-epoch=50-val_loss=8.94-val_center=3.69.ckpt' \
    > logs/test_factory_mvdet_noreid.log 2>&1


CUDA_VISIBLE_DEVICES=1 python -u train_con.py test \
    -c configs/d_factory.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_softmask.yml \
    -c configs/hyp/h_test_tracking_notemp@a0.5_lapjv@0.5_0.75.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/data/namgi/logs/MCMOT' \
    --trainer.logger.version 'test/factory/mvdet/tracking_notemp@a0.5_lapjv@0.5_0.75' \
    --ckpt '/131_data/namgi/logs/MCMOT/lightning_logs/train/factory07/mvdet/simclr@t0.03_60e/checkpoints/model-epoch=50-val_loss=8.94-val_center=3.69.ckpt' \
    > logs/test_factory_mvdet_tracking_notemp@a0.5_lapjv@0.5_0.75.log 2>&1