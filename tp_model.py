from torch import nn
from transformers import ViTModel
from transformers.models.vit import ViTForImageClassification

from vish.constants import IMG_SIZE, VIT_PRETRAINED_MODEL_2
from vish.model.tp.dual import TPDualVit
from vish.model.tp.modified import TPDualModifiedVit
from vish.model.tp.tp_vit import TPVitImageClassification

cifar10_model_params = {
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
    "p_dropout": 0.0,
    "qkv_bias": True,
    "pwff_bias": True,
    "clf_head_bias": True,
    "conv_bias": True,
}

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

imagenet1k_model_params = {
    "img_height": IMG_SIZE,
    "img_width": IMG_SIZE,
    "img_in_channels": 3,
    "patch_dim": 16,
    "emb_dim": 768,
    "num_layers": 12,
    "num_attention_heads": 12,
    "pwff_hidden_dim": 3072,
    "num_classification_heads": 1,
    "mlp_outputs_list": (1000,),
    "p_dropout": 0.0,
    "qkv_bias": True,
    "pwff_bias": True,
    "clf_head_bias": True,
    "conv_bias": True,
}

# Common
# broad_vitForImageClassification = ViTForImageClassification.from_pretrained(
#     VIT_PRETRAINED_MODEL_2
# )
# broad_vitForImageClassification.classifier = nn.Identity()
broad_vitmodel = ViTModel.from_pretrained(VIT_PRETRAINED_MODEL_2)

# cifar10_fine_model = TPVitImageClassification(**cifar10_model_params)
# cifar100_fine_model = TPVitImageClassification(**cifar100_model_params)
imagenet1k_fine_model = TPVitImageClassification(**imagenet1k_model_params)

# TPDualVit
# TP_MODEL_CIFAR100 = TPDualVit(
#     fine_model=cifar100_fine_model, broad_model=broad_vitForImageClassification
# )
# TP_MODEL_CIFAR10 = TPDualVit(
#     fine_model=cifar10_fine_model, broad_model=broad_vitForImageClassification
# )

# # TP Modified Dual Vit
# TP_MODEL_MODIFIED_CIFAR10 = TPDualModifiedVit(
#     fine_model=cifar10_fine_model,
#     broad_model=broad_vitmodel,
#     debug=False,
# )

# TP_MODEL_MODIFIED_CIFAR100 = TPDualModifiedVit(
#     fine_model=cifar100_fine_model,
#     broad_model=broad_vitmodel,
#     debug=False,
# )

TP_MODEL_MODIFIED_IMAGENET1K = TPDualModifiedVit(
    fine_model=imagenet1k_fine_model,
    broad_model=broad_vitmodel,
    debug=False,
)

