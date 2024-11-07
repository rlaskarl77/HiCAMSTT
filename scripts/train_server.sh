python -u train_con.py fit \                       
    -c configs/t_fit_multi_gpu.yml \
    -c configs/d_hdc.yml \
    -c configs/m_mvdet.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir 'reid/train' \
    --trainer.logger.version 'mvdet_res18_hdc-s124t7_simclr_t1' \
    --model.temperature 1.0 \
    > logs/train_mvdet_res18_hdc-s124t7_simclr_t1.log 2>&1