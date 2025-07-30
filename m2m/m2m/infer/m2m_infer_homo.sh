python m2m/m2m/src/inference_m2m.py --infer_real --img_root=dataset/training/homo/img/npy \
    --sensor_types Array-I Circle-I Diamond-I Array-II Circle-II \
    --model_path=checkpoints/m2m/homo/model_11501.pkl \
    --output_dir=infer/homo \
    --dataloader_num_workers=8 --batch_size=1 --save_type=jpg


