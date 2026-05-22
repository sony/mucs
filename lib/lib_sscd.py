import sys, os
import torch
from torchvision import transforms

# From https://github.com/facebookresearch/sscd-copy-detection
TORCHSCRIPT_MODEL_FN = "sscd_disc_mixup.torchscript.pt"

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
transform = transforms.Compose(
    [
        transforms.Resize([320, 320]),
        normalize,
    ]
)


def get_model(device="cpu", cache_dir=None):
    if cache_dir is None:
        fn = TORCHSCRIPT_MODEL_FN
    else:
        fn = os.path.join(cache_dir, TORCHSCRIPT_MODEL_FN)
    model = torch.jit.load(fn)
    model = model.to(device)
    model.eval()
    return model


def get_features(model, data):
    assert data.max() <= 1 and data.min() >= 0
    return model(transform(data))
