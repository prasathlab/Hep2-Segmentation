
#!/bin/bash 
#BSUB -W 80:00
#BSUB -n 24
#BSUB -q surya
#BSUB -P surya
#BSUB -gpu "num=1"
#BSUB -M 128000
#BSUB -e /data/aronow/Balaji_Iyer/Projects/LSTM_Unet/ISIC_2018/Results/logs/%J.err
#BSUB -o /data/aronow/Balaji_Iyer/Projects/LSTM_Unet/ISIC_2018/Results/logs/%J.out

source /usr/local/anaconda3-2020/etc/profile.d/conda.sh
conda activate tf-gpu

res_dir=/data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/Results
code_dir=/data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/src

#First call the train model script
cd ${code_dir}
python run_train.py --base_res_dir ${res_dir} --train_model 





