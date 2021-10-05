
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

res_dir=/data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/Results/New_Models
code_dir=/data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/Hep2-Segmentation

yaml_path="/data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/Hep2-Segmentation/configs/pspnet_NPT_FT/pspnet_NPT_FT.yaml"

#First call the train model script
cd ${code_dir}
python run_train.py --yaml ${yaml_path} --base_res_dir ${res_dir} --train_model




