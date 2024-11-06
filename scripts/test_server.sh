CUDA_VISIBLE_DEVICES=0 python -u test_con.py test \
    -c configs/d_hdc.yml \
    -c configs/m_mvdet.yml \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.save_dir 'reid/test' \
    --trainer.logger.version 'mvdet_res18_hdc-s124t7_simclr_t1_tracking_a@10_inf' \
    --model.temperature 1.0 \
    --ckpt reid/train/lightning_logs/mvdet_res18_hdc-s124t7_simclr_t1/checkpoints/model-epoch=39-val_loss=14.73-val_center=8.85.ckpt \
    > logs/test_contrastive_mvdet_res18_hdc-s124t7_simclr_t1_tracking_a@10_inf.log 2>&1