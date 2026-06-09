import timm
import torch.nn as nn


def get_model(config):
    model = timm.create_model(
        config["model"]["name"],
        pretrained=config["model"]["pretrained"],
        num_classes=config["model"]["num_classes"],
    )
    return model
