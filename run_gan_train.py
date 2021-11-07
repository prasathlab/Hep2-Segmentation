
#from hep2_gan_app_pytorch import HEP2_GAN_app
from hep2_gan_app_keras import HEP2_GAN_Keras_App
import os
import pandas as pd
import sys
from arg_parser import get_parser
import pdb


#yaml_file = r"/data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/Hep2-Segmentation/configs/exp1_cluster_res_unet.yaml"

if __name__ == '__main__':
    sys_argv = sys.argv[1:]
    parser = get_parser()

    args = parser.parse_args(sys_argv)

    yaml_file = args.yaml_file
    print(f"yaml_file path = {yaml_file}")
    #cfg = read_yaml(yaml_file)


    hep2 = HEP2_GAN_Keras_App(yaml_file, args)
    print("Hep2 Initialization Done")


    hep2.train_model()
    print("Model Training Done")

    hep2.model_predict_by_patches()
    print("Model Prediction Done")

    hep2.evaluate_model()
    print("Model Evaluation Done")

    hep2.visualize_model()
    print("Model Visualization Done")

    #To run the model on the entire dataset
    print("Running Test for All 1008 Images")
    hep2.test_dir = os.path.join(hep2.exp_dir, "All_1008")
    os.makedirs(hep2.test_dir, exist_ok=True)

    all_data = r"/data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/raw_data/all_paired_names.tsv"
    hep2.test_df = pd.read_csv(all_data, sep="\t", index_col=0)

    print("Running Model Prediction for All 1008 Images")
    hep2.model_predict_by_patches()

    print("Running Model Evaluation for All 1008 Images")
    hep2.evaluate_model()
    print("All Tasks Done")
