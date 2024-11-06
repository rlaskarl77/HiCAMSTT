python -u train_con.py fit \
    -c configs/t_fit_multi_gpu.yml \
    -c configs/d_hdc.yml \
    -c configs/m_mvdet.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir 'reid/train' \
    --trainer.logger.version 'mvdet_res18_hdc_simclr_t1' \
    --model.temperature 1.0 \
    > logs/train_mvdet_res18_hdc_simclr_t1.log 2>&1 &

python -u train_con.py fit \
    -c configs/t_fit_multi_gpu.yml \
    -c configs/d_hdc.yml \
    -c configs/m_mvdet.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir 'reid/train_test' \
    --trainer.logger.version 'test_train_image' \
    --model.temperature 1.0 \
    > logs/test_train_image.log 2>&1 &

python -u train_con.py fit \
    -c configs/t_fit_multi_gpu.yml \
    -c configs/d_hdc.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_default.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir 'reid/train_test_cvpr' \
    --trainer.logger.version 'train_test_cvpr' \
    --model.temperature 1.0 \
    > logs/train_test_cvpr.log 2>&1

python -u train_con.py fit \
    -c configs/t_fit_multi_gpu.yml \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_hardmask.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'train_test_cvpr' \
    > logs/cvpr2025/train_test_cvpr.log 2>&1

python -u train_con.py fit \
    -c configs/t_fit_multi_gpu.yml \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_simclr@t0.03_pseudolabel_hardmask.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'train_test_cvpr' \
    > logs/cvpr2025/train_test_cvpr.log 2>&1

cd /131_data/namgi/TrackTacular
cd WorldTrack
python -u train_con.py fit\
    -c configs/t_fit_multi_gpu.yml \
    -c configs/d_wildtrack_server.yml \
    -c configs/m_mvdet.yml \
    -c configs/hyp/h_baseline.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir '/131_data/namgi/logs/MCMOT' \
    --trainer.logger.version 'train/wildtrack/mvdet/baseline' \
    > logs/cvpr2025/train_wildtrack_mvdet_baseline.log 2>&1