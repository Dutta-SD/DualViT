from vish.constants import IMG_SIZE, VIT_PRETRAINED_MODEL_2
from vish.model.tp.dual import TPDualVit
from vish.model.tp.tp_vit import TPVitImageClassification
from transformers.models.vit import ViTForImageClassification
from torch import nn


model_params = {
    "img_height": 224,
    "img_width": 224,
    "img_in_channels": 3,
    "patch_dim": 16,  # reduced patch size
    "emb_dim": 768,
    "num_layers": 12,
    "num_attention_heads": 12,
    "pwff_hidden_dim": 3072,
    "num_classification_heads": 1,
    "mlp_outputs_list": (10,),  # Cifar 10 default
    "p_dropout": 0.1,
    "qkv_bias": True,
    "pwff_bias": True,
    "clf_head_bias": True,
    "conv_bias": True,
}

# MODEL CONFIGURATION
fine_model = TPVitImageClassification(**model_params)
broad_model = ViTForImageClassification.from_pretrained(VIT_PRETRAINED_MODEL_2)

broad_model.classifier = nn.Identity()

TP_MODEL = TPDualVit(fine_model, broad_model)

cifar100_model_params = {
    "img_height": IMG_SIZE,
    "img_width": IMG_SIZE,
    "img_in_channels": 3,
    "patch_dim": 16,  # reduced patch size
    "emb_dim": 768,
    "num_layers": 12,
    "num_attention_heads": 12,
    "pwff_hidden_dim": 3072,
    "num_classification_heads": 1,
    "mlp_outputs_list": (100,),  # Cifar 10 default
    "p_dropout": 0.0,
    "qkv_bias": True,
    "pwff_bias": True,
    "clf_head_bias": True,
    "conv_bias": True,
}

fine_model_cifar100 = TPVitImageClassification(**cifar100_model_params)
TP_MODEL_CIFAR100 = TPDualVit(fine_model_cifar100, broad_model)