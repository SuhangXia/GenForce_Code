#### Example:

- task (*name of the task*): Array-I_Array-II (*Transfer from Array-I to Array-II*)

- src_img (*generated imgs used to train model.*): infer/homo/Array-I_Array-II/01_13_07_09

- src_force (*source forces used to train model.*): dataset/homo/force/Array-I

- tar_img (*target imgs used to predict the forces in inference stage by using the trained model*): dataset/homo/image/npy/Array-II

- tar_force (*target groundtruth forces in inference stage to calculate the errors*): dataset/homo/force/Array-II

- min_max (*min-max file for normalization*): force/config/min_max.json

- checkpoint (*checkpoints*): infer/force/Array-II/Array-I_Array-II/20250211_205151/epoch_40.pth

- save_dir (*save dir for checkpoints*): infer/force/Array-II/{}

- unseen (*indenters chosen as unseen, not included in training*): ["sphere_s", "triangle", "pacman", "cone", "wave", "torus"]
test_unseen: False

- \# Train (used in training, comment in infer)
- train (whether to train the model): True

- train_use_checkpoint (whether to use checkpoint. False: train from scratch): True    

- draw_force (whether to draw the force prediction results): True

- draw_use_checkpoint (whether to draw the force prediction results using the checkpoint. Only select true when not training but directly draw the figure with checkpoints): False 
     
- \# Infer (used in infer. comment in training stage)

- \# train: False

- \# train_use_checkpoint: True                 

- \# draw_force: True

- \# draw_use_checkpoint: True               

- num_workers: 16
- batch_size: 2
- n_epoch: 40
- early_stop: 20
- lr (set as 1e-1 for training. fine tune:1e-3): 1e-1  
- seed: 0

