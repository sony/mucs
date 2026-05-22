import os
import argparse
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.func import vmap


# compute loss for a single sample at a time
def compute_loss(model, weights, buffers, loss_configs, x, y):
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)

    sigma_P_mean, sigma_P_std = loss_configs.sigma_P
    sigma_data = loss_configs.sigma_data

    # Sample a random sigma for each image
    s = torch.randn([len(x), 1, 1, 1], device=x.device)
    sigma = (sigma_P_std * s + sigma_P_mean).exp()
    noise = sigma * torch.randn_like(x)
    noisy_x = x + noise

    xhat = torch.func.functional_call(
        model, (weights, buffers), args=(noisy_x, sigma.view(-1), y)
    )
    weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
    loss = (weight * (xhat - x).pow(2)).mean()
    return loss


# defines a function that takes in data, model, and noise scheduler and outputs the gradients using vmap
def grads_loss(x, y, model, grads_fn, func_weights, func_buffers, loss_configs):
    # compute the gradients
    gs = vmap(grads_fn, in_dims=(None, None, None, None, 0, 0), randomness="different")(
        model, func_weights, func_buffers, loss_configs, x, y
    )

    return gs


# iterate over the dataset and compute the gradients for each image to get the fisher information
def estimate_fisher(model, data_loader, loss_configs, num_epochs=5):
    func_weights = dict(model.named_parameters())
    func_buffers = dict(model.named_buffers())
    # Function to compute gradients with respect to its inputs
    grads_fn = torch.func.grad(compute_loss, has_aux=False, argnums=1)
    # print(func_weights, func_buffers)

    sample_iter = 0
    fisher_accum = None
    accum_count = 0
    print("Gathering fisher information...")
    for epoch in range(num_epochs):
        pbar = tqdm(data_loader)
        for batch in pbar:
            if len(batch) >= 3:
                idx, x, y = batch[:3]
            else:
                raise NotImplementedError

            current_bs = len(x)
            sample_iter += current_bs

            x = x.cuda()
            y = y.cuda()

            # compute the gradients
            grad = grads_loss(
                x, y, model, grads_fn, func_weights, func_buffers, loss_configs
            )
            # print(grad)
            with torch.no_grad():
                fisher_avg = {
                    key: grad[key].double().square().mean(dim=0).detach()
                    for key in grad.keys()
                }
                if fisher_accum is None:
                    fisher_accum = fisher_avg
                else:
                    for key in fisher_accum.keys():
                        fisher_accum[key] += fisher_avg[key]
            accum_count += 1
            pbar.set_description(f"Epoch {epoch + 1} / {num_epochs}")
        pbar.close()

    # average the fisher information
    with torch.no_grad():
        fisher_diagonals = {
            key: (fisher_accum[key] / accum_count) for key in fisher_accum.keys()
        }

    print("Fisher info estimation done...")
    return fisher_diagonals


def collect_loss(
    model,
    data_loader,
    model_configs,
    time_samples=20,
    avg_timesteps=False,
    init_random_seed=0,
    device="cuda",
):
    """
    Collect EDM loss for each training sample.
    IMPORTANT: Fix batch_size, time_samples, and init_random_seed to retain same random noise for each train point.
    :param model: model
    :param data_loader: data loader
    :param noise_scheduler: noise scheduler
    :param time_samples: number to uniform subsample the timesteps
    :param avg_timesteps: whether to average over timesteps
    :param random_seed: random seed
    :return: loss for each training sample (shape: [num_train, time_samples] or [num_train])
    """

    step_indices = torch.arange(time_samples, device=device)
    sigma_max = model_configs.sampler.sigma_max
    sigma_min = model_configs.sampler.sigma_min
    rho = model_configs.sampler.rho
    sigma_data = model_configs.loss.sigma_data

    timesteps = torch.tensor(
        (
            sigma_max ** (1 / rho)
            + step_indices
            / (time_samples - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        )
        ** rho
    )
    print("Sigma values for computing loss: ", timesteps.tolist())

    losses = []
    batch_id = 0
    with torch.inference_mode():
        for batch in tqdm(data_loader):
            if len(batch) >= 3:
                idx, x, y = batch[:3]
            else:
                raise NotImplementedError

            x = x.to(device)
            current_bs = x.shape[0]
            x = (
                x.unsqueeze(1)
                .expand(-1, time_samples, -1, -1, -1)
                .reshape(-1, *x.shape[-3:])
            )
            # print("Input min/max:", x.min().item(), x.max().item())

            y = y.to(device)
            y = y.unsqueeze(1).expand(-1, time_samples, -1).reshape(-1, *y.shape[-1:])

            noise_seed = init_random_seed + batch_id
            generator = torch.Generator(device="cuda")
            generator.manual_seed(noise_seed)

            # with torch.no_grad():
            # multiple values of sigma for each sample in every batch
            sigma = timesteps.unsqueeze(0).T.repeat(1, current_bs).view(-1)
            sigma = sigma.reshape(current_bs * time_samples, 1, 1, 1)
            noise = sigma * torch.randn(
                x.shape, device=x.device, dtype=x.dtype, generator=generator
            )
            noisy_x = x + noise

            xhat = model(noisy_x, sigma.view(-1), y)
            # print("model output nans: ",torch.isnan(xhat).any())

            weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
            loss = (weight * (xhat - x).pow(2)).mean(dim=[1, 2, 3])
            loss = loss.reshape(current_bs, time_samples)

            if avg_timesteps:
                loss = loss.mean(axis=1)
            loss = loss.cpu().numpy()
            # print("Loss values: ",loss)
            losses.append(loss)
            # print("Loss nans: ",torch.isnan(xhat).any())

            batch_id += 1

    losses = np.concatenate(losses, axis=0)
    return losses


def get_optimizer(lr, model, optimizer_name, unlearn_params=None):
    # Select parameters to optimize
    params_to_optimize = []
    param_names_to_optimize = []
    for name, param in model.named_parameters():
        if unlearn_params is not None and unlearn_params in name:
            params_to_optimize.append(param)
            param_names_to_optimize.append(name)
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    # Select optimizer based on optimizer_name
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(params_to_optimize, lr=lr, weight_decay=0)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(params_to_optimize, lr=lr, weight_decay=0)
    elif optimizer_name is None:
        optimizer = None
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")

    return optimizer, param_names_to_optimize


def print_param_info(model):
    train_param_count = 0
    all_param_count = 0
    train_param_names = []
    all_param_names = []

    # Count trainable and all parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            train_param_names.append(name)
            train_param_count += param.numel()
        all_param_names.append(name)
        all_param_count += param.numel()

    # print(
    #     "####################\n"
    #     "# Trainable params #\n"
    #     "####################\n",
    #     "\n".join(train_param_names)
    # )

    print(f"All parameter count: {all_param_count}")
    print(f"Trainable parameter count: {train_param_count}")


class GenDataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
        # self.transform = torchvision.transforms.Normalize(
        #     mean=0.5,
        #     std=0.5,
        #     inplace=True,
        # )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return idx, self.samples[idx], self.labels[idx]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())  # if p.requires_grad)
