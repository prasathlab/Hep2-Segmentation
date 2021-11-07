
#!/bin/bash 
#BSUB -W 80:00
#BSUB -n 24
#BSUB -q surya
#BSUB -P surya
#BSUB -gpu "num=1"
#BSUB -M 200000
#BSUB -e /data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/logs/%J.err
#BSUB -o /data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/logs/%J.out

source /usr/local/anaconda3-2020/etc/profile.d/conda.sh
conda activate tf-gpu

res_dir=/data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/Results/With_Rand_Aug
code_dir=/data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/Hep2-Segmentation

yaml_path="/data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/Hep2-Segmentation/rand_aug_configs/rand_aug_resnet50_unet_NPT_FT/rand_aug_resnet50_unet_NPT_FT.yaml"

#First call the train model script
cd ${code_dir}
weights_path="/data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/Results/New_Models/resnet50_unet_NPT_FT_2021-09-12_03-56-29/Models/Final_Model/best_model.hdf5"
python run_train.py --yaml ${yaml_path} --base_res_dir ${res_dir} --train_model --model_weights ${weights_path}




