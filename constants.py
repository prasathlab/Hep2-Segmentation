from unet_models.vanilla_unet import get_vanilla_unet_model
from unet_models.res_unet import get_residual_unet_model
from unet_models.dense_unet import get_dense_unet_model
from fcn_models.fcn import *
from fcn_models.segnet import *
from fcn_models.unet import *
from fcn_models.pspnet import *
from fcn_models.hr_net import *
from GANs_keras.gan_discriminator_models import *

Supported_Archs = {"Vanilla_U-Net": get_vanilla_unet_model,
                   "Res_U-Net": get_residual_unet_model,
                   "Dense_U-Net": get_dense_unet_model,
                   "fcn_8": fcn_8,
                   "fcn_8_vgg": fcn_8_vgg,
                   "fcn_32": fcn_32,
                   "fcn_32_vgg": fcn_32_vgg,
                   "fcn_8_resnet50": fcn_8_resnet50,
                   "fcn_32_resnet50": fcn_32_resnet50,
                   "fcn_8_mobilenet": fcn_8_mobilenet,
                   "fcn_32_mobilenet": fcn_32_mobilenet,
                   "segnet": segnet,
                   "vgg_segnet": vgg_segnet,
                   "resnet50_segnet": resnet50_segnet,
                   "mobilenet_segnet": mobilenet_segnet,
                   "unet":unet,
                   "vgg_unet": vgg_unet,
                   "resnet50_unet": resnet50_unet,
                   "mobilenet_unet": mobilenet_unet,
                   "pspnet": pspnet,
                   "vgg_pspnet": vgg_pspnet,
                   "resnet50_pspnet": resnet50_pspnet,
                   "hr_net": hr_net,
                   "basic_discriminator": get_basic_discriminator,
                   "class_conditional_discriminator": get_class_cond_discriminator,
                   "patchGAN": get_patchgan_discriminator

                   }

Supported_Backbones = ["vgg", "resnet50", "mobilenet"]