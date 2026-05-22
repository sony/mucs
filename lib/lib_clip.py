import sys
import torch
import open_clip
from torchvision.transforms import ToPILImage

to_pil = ToPILImage()


def get_model(model_device="cpu", cache_dir=None):
    model, _, preproc = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion2b_s32b_b82k", cache_dir=cache_dir
    )
    model = model.to(model_device)
    model.eval()
    return model, preproc


def get_features(model, preproc, data):
    assert data.max() <= 1 and data.min() >= 0
    device = data.device
    data = data.cpu()
    images = [to_pil(img) for img in data]  # list of PIL images
    x = torch.stack([preproc(img) for img in images])
    x = x.float().to(device)
    return model.encode_image(x)
