# GenForce_Code

The code for paper:

**Training Tactile Sensors to Learn Force Sensing from Each Other**
(published in Nature Communiations 2026)

[Project page](https://zhuochenn.github.io/genforce-project/). 

Zhuo Chen1, Ni Ou1, Xuyang Zhang1, Zhiyuan Wu1, Yongqiang Zhao1, Yupeng Wang1, Nathan Lepora2, Lorenzo Jamone3, Jiankang Deng4, Shan Luo1

- 1 King’s College London, London, United Kingdom.
- 2 University of Bristol, Bristol, United Kingdom.
- 3 University College London, London, United Kingdom.
- 4 Imperial College London, London, United Kingdom.


![image](assets/cover.jpg)

# Overview
A framework, GenForce, that enables transferable force sensing across tactile sensors. GenForce unifies tactile signals into shared marker representations, analogous to cortical sensory encoding, allowing force prediction models trained on one sensor to be transferred to others without the need for exhaustive force data collection.  More Details can be found in our paper.

The GenForce model contains two modules:

* **Marker-to-marker translation model** ([m2m](/m2m)). The m2m module is available to transfer the deformation across arbitrary marker representations. The first step is to train the model to bridge the source sensors and the target sensors using the m2m model.This end-to-end model enables direct translation of marker-based images from source images to generated images with the image style of target sensors while preserving the deformation from source sensors. This model is based on image-conditioned diffusion model, which is scalable with the increasing types of marker images. It can achieve many-to-many translation within one model and the generated image styles are chosen by the conditioned reference images.

* **Force prediction model** ([force](/force)). After training m2m model, we can transfer all of the marker images with force labels from the old sensor to new sensors (get the generated images), allowing to use the transferred marker images and the existing labels to train force prediciton models to target sensors.

# Getting Started
## Environment
We test our code on NVIDIA A100, 80GB memory. The mininum memory to run our code is 8GB with bathsize=1. Reduce batch_size if out of memory; 
- Install required denpendencied using our conda env file
```
conda env create -f environment.yaml
```
- Activate the conda environment
```
conda activate genforce
```

## 0. Data Collection

### Simulation for marker deformation

> Marker simulation is used to generate any marker deformation before real-world data collection and is helpful to get a pretrained M2M model. 

Tested in ubuntu 20.0. 

- Install `pcl-tools`, `blender`
```
sudo apt install pcl-tools
sudo snap install blender --channel=3.3lts/stable --classic
```
- Elastomer deformation 
```
python sim/deformation/1_stl2npy.py  # generate .npy file for indenter used in simulation
python sim/deformation/2_deformation.py # get elastomer deformation with different indenters and contact positions
python sim/deformation/3_npz2stl.py # transfer .npz file to stl file for rendering marker images
```
> The input files are in [input](sim/assets/indenters/input), and output files are generated in sim/assets/indenters/output after running the code.

> For test the code only, no need to run 2_deformation.py to the end. Get some npz files can continue run step3-4 to see the results.

> If want to get same amounts of data similar to our dataset, step 2_deformation.py
need to finish.

- Marker rendering
```
blender -b --python sim/marker/4_render.py
```
> Our designed marker patterns are in [marker_pattern](sim/marker/marker_pattern). Generated marker images are in [marker](sim/assets/marker) after running the code.

> Can replace the marker patterns with real-world reference tactile images, such as from GelSight, to get realistic deformation.

### Real-world data collection (In real sensors)

Setup: Robot arm (or any 3DoF moving platform), [indenters](sim/assets/indenters/input/stl), tactile sensors

> We use ur5e + ATI nano 17 in our paper, see paper and code [data_collection](data_collection/components/).

Step 1. Collect the data by referring to the trajectory and code in our paper. If want to use material compensation, force-depth curved needed to be measured.

Step 2. Marker segmentation, see code [segmentation](data_collection/marker_seg)

#### Our Dataset

> To test the genforce model without data collection, our dataset (simulation and real-world) and checkpoints can be downloaded from [Dataset](https://emckclac-my.sharepoint.com/:f:/g/personal/k23058530_kcl_ac_uk/IgCxLaGHRPTkQpT6kIrZSYueAbJGFahvqJB5RADacqodp7A). 



> Download and unzip the dataset (/training.zip, /training/material_compensation.zip, and /img_gen/img_gen.zip) into dataset/, checkpoints (/training/checkpoints/m2m_checkpoint.zip, force_checkpoint.zip) into checkpoints/.

> The final folder structure can refer to [dataset_folder](assets/folder_struc_dataset.txt)

> For training the model only, no need to use raw_data as we have converted some of the raw images into marker images in /training.

> All the marker images are saved with np.packbits() to reduce memory cost. To see the image use
```
from PIL import Image
import numpy as np

image_path = "dataset/homo/img/npy/Array-I_ref.npy" # modify the path
loaded_image = np.load(image_path)
loaded_image = np.unpackbits(loaded_image).reshape((480,640))*255
loaded_image = Image.fromarray(loaded_image.astype(np.uint8)).convert('RGB')
loaded_image.show()
```

## 1. Training for maker-to-marker translation

Step1 to Step3 are the process we trained our model for the experiments in our paper.

For utilizing the model on other sensors, users just need to collect location paired images as the trajectories used in our paper and finetune the model with the checkpoints in Step 3.

> To successfully run below code, change the --dataset_folder argument in each .sh file within [folder](m2m/m2m) with your dataset location. Explaination for the key arguments of .sh file in M2M, find [here](assets/arugment_m2m.txt)

> Run the first time, you may need to configure your wandb account.

> Note that, in training stage, we only use four images, such as images listed in dataset/training/homo/img/npy/Circle-II/moon/1/last.csv in each contact point folder(dataset/training/homo/img/npy/Circle-II/moon/1). While in inference stage, we convert them all. Thus, a majority of images are unseen in the training stage.

### Step1. Training for marker encoder
- To train the maker-to-marker translation model, we first train a marker encoder for marker feature extraction. 
```
sh m2m/vae/marker_encoder.sh
```
> skip this step if using our checkpoint in the next steps: checkpoints/m2m/vae/model_70000.pth

> You can change the arguments in the main function of [vae](m2m/vae/src/marker_encdoer.py) for training parameters.

### Step2. Pretraining with simulated data
- We freeze marker encoder for the image condition and pretrain the m2m model with simulated data. 

> change the ---output_dir, --dataset_folder and --pretrained_ref_encoder in [.sh](m2m/m2m/m2m_sim.sh) with your path

```
sh m2m/m2m/m2m_sim.sh
```
> If training from scratch, comment --pretrained_model_name_or_path and delete '\' in the last line

### Step3. Fintuning with real-world data 
#### homogeneous translation
- Finetuning the m2m model with homogeneous data. 
```
sh m2m/m2m/infer/m2m_homo.sh
```
#### material softness effect
- Finetuning the m2m model with material softness effect data. 
```
sh m2m/m2m/infer/m2m_modulus.sh
```
#### homogeneous translation
- Finetuning the m2m model with homogeneous data. 
```
sh m2m/m2m/infer/m2m_heter.sh
```
## 2. Inference for maker-to-marker translation

Upon training m2m model, we can convert all the images with force labels from the source sensors to target sensors.

> change the from --save_type=npy to --save_type=jpg can directly get images with .jpg format
- simulated data. 
```
sh m2m/m2m/infer/m2m_infer_sim.sh
```
- homogeneous translation. 
```
sh m2m/m2m/infer/m2m_infer_homo.sh
```
- material softness effect. 
```
sh m2m/m2m/infer/m2m_infer_modulus.sh
```
- heterogeneous translation. 
```
sh m2m/m2m/infer/m2m_infer_heter.sh
```
## 3. Training for force prediciton models
After inference, we get the generated images and force labels from the source sensors. We can use those data to train the force prediction model for each target sensor.

> We provide some generated images and checkpoints in our dataset. Full generated images can use above inference to get other images if interested in.

- homogeneous translation (run bash files in [Folder](force/scripts/homo))

For example, to tranfer to Array-II
```
sh force/scripts/homo/seen/m2m/Array-II/Array-II.sh
```
> Change the path in the [.yaml](force/scripts/homo/seen/m2m/Array-II) file to your path. See [Argument](/assets/argument_force.md) to config the .yaml file.

- material softness effect. 
For example, to tranfer to ratio8 without compensation ([wo_com](force/wo_com))
```
sh force/scripts/modulus/train/wo_com/ratio8/8.sh
```
- heterogeneous translation. 
For example, to train all heterogeneous sensors with compensation [com](force/com/com_hetero), starting depth 0, correction weight 0.5
```
sh force/scripts/hetero/seen/com/grid/0_0.5.sh
```
> Folers named with /com include config file for material compensation; /wo_com include config files without compensation

> Change the path in the [.yaml](force/scripts/hetero/seen/com/gelsight/tactip_gelsight.yaml) file to your path. 

> argument: --src_depth means the path for contact depth files ;- -modulus means the material prior, saved with coefficient such as dataset/training/material_compensation/hetero/cofficient and dataset/training/material_compensation/modulus/cofficients/modulus. Note that, both files are in the dataset not in the repo.

> Change the arg `modulus` in [.yaml](force/scripts/hetero/seen/com/gelsight/tactip_gelsight.yaml) as dataset/training/material_compensatin/xxxx(hetero)/cofficient. 

# Citation 

# Contact
Any questions, feel free to reach out to:

Zhuo Chen: *zhuo.7.chen@kcl.ac.uk*
