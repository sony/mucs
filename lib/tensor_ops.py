import sys
import torch, math
from einops import rearrange

###################################################################################################


def tensor_quantile(x, q, dim=-1, keepdim=False):
    assert x.ndim == q.ndim
    qn = (q.clamp(min=0, max=1) * (x.size(dim) - 1)).round().long()
    sx = x.sort(dim=dim)[0]
    xq = torch.gather(sx, dim, qn)
    if keepdim:
        return xq
    return xq.squeeze(dim)


###################################################################################################


def debug_inf_nan(ten, txt):
    if torch.isnan(ten).float().sum() > 0:
        print()
        print("nan " + txt)
        sys.exit()
    if torch.isinf(ten).float().sum() > 0:
        print()
        print("inf " + txt)
        sys.exit()


###################################################################################################


def covariance(x, eps=1e-6):
    xx = x - x.mean(0, keepdim=True)
    cov = torch.matmul(xx.T, xx) / (len(xx) - 1)
    weight = torch.triu(torch.ones_like(cov), diagonal=1)
    cov = (weight * cov.pow(2)).sum() / (weight.sum() + eps)
    return cov


###################################################################################################


def roughly_equal(x, y, tol=1e-6):
    return (x - y).abs() < tol


###################################################################################################


def pairwise_euclidean_distance_matrix(x, y, squared=False, eps=1e-6):
    squared_x = x.pow(2).sum(1).view(-1, 1)
    squared_y = y.pow(2).sum(1).view(1, -1)
    dot_product = torch.mm(x, y.t())
    distance_matrix = squared_x - 2 * dot_product + squared_y
    # get rid of negative distances due to numerical instabilities
    distance_matrix[distance_matrix <= 0] = 0
    if not squared:
        # handle numerical stability
        # derivative of the square root operation applied to 0 is infinite
        # we need to handle by setting any 0 to eps
        mask = (distance_matrix == 0.0).type_as(distance_matrix)
        # use this mask to set indices with a value of 0 to eps
        distance_matrix += mask * eps
        # now it is safe to get the square root
        distance_matrix = torch.sqrt(distance_matrix)
        # undo the trick for numerical stability
        distance_matrix *= 1.0 - mask
    return distance_matrix


def pairwise_distance_matrix(x, y, mode="fro", p=2, eps=1e-6):
    assert x.ndim == y.ndim and x.ndim <= 2
    assert mode in (
        "fro",
        "nfro",
        "euc",
        "neuc",
        "sqeuc",
        "nsqeuc",
        "cos",
        "cossim",
        "dot",
        "dotsim",
    )
    # Prepare
    if x.ndim == 1:
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
    if mode == "euc" or mode == "neuc":
        p = 2
    # Choose mode of operation
    if mode in ("fro", "nfro", "euc", "neuc"):
        dist = torch.cdist(x.unsqueeze(0), y.unsqueeze(0), p=p).squeeze(0)
        if mode == "nfro" or mode == "neuc":
            dist = dist / (x.size(-1) ** (1 / p))
    elif mode in ("sqeuc", "nsqeuc"):
        dist = pairwise_euclidean_distance_matrix(x, y, squared=True)
        if mode == "nsqeuc":
            dist = dist / x.size(-1)
    elif mode in ("cos", "cossim", "dot", "dotsim"):
        if mode == "cos" or mode == "cossim":
            x = x / (torch.norm(x, dim=-1, keepdim=True) + eps)
            y = y / (torch.norm(y, dim=-1, keepdim=True) + eps)
        dist = torch.matmul(x, y.T)
        if mode == "cos" or mode == "dot":
            dist = 1 - dist
    else:
        raise NotImplementedError
    return dist


###################################################################################################


def cosine_similarity(x, y, dim=-1, keepdim=False):
    x = torch.nn.functional.normalize(x, dim=dim)
    y = torch.nn.functional.normalize(y, dim=dim)
    return (x * y).sum(dim=dim, keepdim=keepdim)


###################################################################################################


def euclidean_distance(x, y, dim=-1, keepdim=False, averaged=False, squared=False):
    return l2norm(x - y, dim=dim, keepdim=keepdim, averaged=averaged, squared=squared)


def l2norm(x, dim=-1, keepdim=False, averaged=False, squared=False):
    x = x.pow(2)
    if averaged:
        x = x.mean(dim, keepdim=keepdim)
    else:
        x = x.sum(dim, keepdim=keepdim)
    if squared:
        return x
    return x.sqrt()


###################################################################################################


def zscore(x, dim=-1, eps=1e-9):
    mu = x.mean(dim=dim, keepdim=True)
    sigma = x.std(dim=dim, keepdim=True)
    return (x - mu) / (sigma + eps)


###################################################################################################
