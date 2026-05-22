import sys, os
import torch

from utils import print_utils


class Attributor(object):

    def __init__(
        self,
        conf,
        fabric,
        model,
        method_path,
        device="cpu",
    ):
        self.conf = conf
        self.fabric = fabric
        self.model = model
        self.method_path = method_path
        self.device = device

    def run(
        self,
        dl_train,
        samp_gen_attrib_x,
        samp_gen_attrib_y,
        extra,
        top_k_attrib,
    ):
        progbar = lambda it, desc=None: print_utils.myprogbar(
            it, desc=desc, leave=True, doit=self.fabric.is_global_zero
        )

        with torch.inference_mode():
            all_idx = []

            for batch in progbar(dl_train):
                if len(batch) >= 3:
                    idx, x, y = batch[:3]
                else:
                    raise NotImplementedError
                all_idx.append(idx)
            all_idx = torch.cat(all_idx, dim=0)
            vals = torch.rand(len(samp_gen_attrib_x), len(all_idx))

            # Sort and get correct indices
            ids = torch.argsort(vals, dim=-1, descending=True)
            ids = ids[:, :top_k_attrib]
            ret_idx = []
            for i in range(len(ids)):
                ret_idx.append(all_idx[ids[i]])
            ret_idx = torch.stack(ret_idx, dim=0)
        return ret_idx, all_idx, vals
