from models.vanilla_unet import get_vanilla_unet_model
from models.res_unet import get_residual_unet_model
from models.dense_unet import get_dense_unet_model

Supported_Archs = {"Vanilla_U-Net": get_vanilla_unet_model,
                   "Res_U-Net": get_residual_unet_model,
                   "Dense_U-Net": get_dense_unet_model,

                   }