
from hep2_app import HEP2_app
import pdb


yaml_file = r"/data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/src/configs/exp1_cluster.yaml"

hep2 = HEP2_app(yaml_file)
print("Hep2 Initialization Done")
hep2.train_model()
print("Model Training Done")
hep2.model_predict_by_patches()
print("Model Prediction Done")
hep2.evaluate_model()
print("Model Evaluation Done")
hep2.visualize_model()
print("Model Visualization Done")
print("All Tasks Done")