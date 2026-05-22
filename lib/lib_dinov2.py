import sys, warnings
import torch
from torchvision import transforms


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_model(device="cpu"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval()
    return model.to(device)


def get_features(model, data):
    assert data.max() <= 1 and data.min() >= 0
    return model(transform(data))
