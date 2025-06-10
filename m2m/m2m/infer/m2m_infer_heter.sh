python m2m/m2m/src/inference_m2m.py --infer_real --img_root=dataset/training/hetero/img \
    --sensor_types gelsight uskin tactip\
    --model_path=checkpoints/m2m/hetero/model_25501.pkl \
    --output_dir=infer/hetero \
    --dataloader_num_workers=32 --batch_size=1 \
    --save_type=jpg
