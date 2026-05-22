import sys, os
import torch, math
import time

from utils import pytorch_utils, print_utils


###############################################################################


def get_checkpoint_names(path, prefix="checkpoint"):
    fn_last = os.path.join(path, prefix + "_last.ckpt")
    fn_best = os.path.join(path, prefix + "_best.ckpt")
    fn_epoch = os.path.join(path, prefix + "_$epoch$.ckpt")
    return fn_last, fn_best, fn_epoch


###############################################################################


class Trainer(object):

    def __init__(
        self,
        conf,
        fabric,
        model,
        ema,
        optim,
        sched,
        augment=None,
        limit_batches=None,
        seed_valid=45,
    ):
        # Global stuff
        self.conf = conf.training
        self.all_conf = conf  # for saving checkpoints
        self.fabric = fabric
        self.model = model
        self.model.mark_forward_method("loss")
        self.ema = ema
        if self.ema is not None:
            self.ema.mark_forward_method("loss")
        self.optim = optim
        self.sched = sched
        # Local stuff
        self.augment = augment
        self.limit_batches = limit_batches
        self.seed_valid = seed_valid
        # Utils
        self.progbar = lambda it, desc=None: print_utils.myprogbar(
            it, desc=desc, leave=False, doit=self.fabric.is_global_zero
        )
        self.timer = print_utils.Timer()
        self.current_time = print_utils.get_time_base()  # for printing every xxx

    ###########################################################################

    def run(
        self,
        dl_train,
        dl_valid=None,
        start_epoch=0,
        end_epoch=None,
        cost_best=None,
    ):
        # Prepare
        if dl_valid is None:
            dl_valid = dl_train
        if end_epoch is None:
            end_epoch = self.conf.num_epochs
        if cost_best is None:
            cost_best = torch.inf if self.conf.monitor.mode == "min" else -torch.inf
            if self.conf.optim.sched.startswith("plateau"):
                self.sched.step(cost_best)
        epoch_best = start_epoch
        lr = self.sched.get_last_lr()[0]

        # Epoch loop
        for epoch in range(start_epoch, end_epoch):
            desc = f"{epoch+1:{len(str(end_epoch))}d}/{end_epoch}"

            # Basic logging
            self.fabric.log("hpar/epoch", epoch + 1, step=epoch + 1)
            self.fabric.log("hpar/lr", lr, step=epoch + 1)

            # Train
            logdict_train = self._loop_batches(dl_train, training=True, desc=desc)
            logdict_train.sync_and_mean(self.fabric)
            self.fabric.log_dict(logdict_train.get(prefix="train/"), step=epoch + 1)

            # Valid
            with torch.inference_mode():
                logdict_valid = self._loop_batches(dl_valid, training=False, desc=desc)
                logdict_valid.sync_and_mean(self.fabric)
                self.fabric.log_dict(logdict_valid.get(prefix="valid/"), step=epoch + 1)

            # Get report & check NaN/inf
            tmp = logdict_valid.get(keys=["l_main"])
            tmp["l_main_t"] = logdict_train.get("l_main")
            report = print_utils.report(tmp, desc=f"[{self.timer.time()}] Epoch {desc}")
            for aux in tmp.values():
                if math.isnan(aux) or math.isinf(aux):
                    self.fabric.print(flush=True)
                    return "NaN or inf detected!"

            # Optimizer schedule
            if self.conf.optim.sched.startswith("plateau"):
                self.sched.step(cost_current)
            else:
                self.sched.step()
            new_lr = self.sched.get_last_lr()[0]
            if new_lr != lr:
                if self.conf.optim.sched.startswith("plateau"):
                    report += f"  (lr={new_lr:.1e})"
                lr = new_lr

            # Best cost
            cost_current = logdict_valid.get(self.conf.monitor.quantity)
            is_best = False
            if (self.conf.monitor.mode == "max" and cost_current > cost_best) or (
                self.conf.monitor.mode == "min" and cost_current < cost_best
            ):
                is_best = True
                cost_best = cost_current
                epoch_best = epoch
                report += "  *"

            # Checkpointing
            self.fabric.print("Checkpointing... ", end="", flush=True)
            self._checkpointing(epoch, cost_current, is_best=is_best)

            # Done
            self.fabric.print(report)
            if self.conf.optim.min_lr is not None and lr < self.conf.optim.min_lr:
                return "Minimum lr reached.", cost_best
            if (
                "num_epochs_no_improv" in self.conf
                and self.conf.num_epochs_no_improv is not None
                and epoch - epoch_best >= self.conf.num_epochs_no_improv
            ):
                return "No improvement.", cost_best

        return None, cost_best

    ###########################################################################

    def _loop_batches(self, dataloader, training=False, desc=None):
        # Init
        if training:
            num_updates = self.conf.num_updates_train
            desc = "Train " + desc if desc is not None else None
            model = self.model
            model.train()
        else:
            num_updates = self.conf.num_updates_valid
            desc = "Valid " + desc if desc is not None else None
            model = self.ema if self.ema is not None else self.model
            model.eval()
        logdict = pytorch_utils.LogDict()
        # Loop
        i = 0
        while True:
            if num_updates is not None:
                newdesc = desc + "-" + str(i).zfill(len(str(num_updates - 1)))
            else:
                newdesc = desc
            self.fabric.barrier()
            for batch in self.progbar(dataloader, desc=newdesc):
                # Update model
                logdict = self._batch_update(model, batch, logdict, training=training)
                # EMA
                if training and self.ema is not None:
                    pytorch_utils.ema_update(model, self.ema, decay=self.conf.ema)
                # Print
                self._print_losses(logdict)
                # Exit?
                i += 1
                if num_updates is not None and i >= num_updates:
                    return logdict
                if self.limit_batches is not None and i >= self.limit_batches:
                    return logdict
            # self._print_losses(logdict, force=True)
            if num_updates is None:
                return logdict

    ###########################################################################

    def _batch_update(self, model, batch, logdict, training=False):
        # Prepare data
        if len(batch) >= 3:
            idx, x, y = batch[:3]
        else:
            raise NotImplementedError
        a = None
        if training:
            rng = None
        else:
            rng = pytorch_utils.StackedRandomGenerator(
                seeds=self.seed_valid + idx, device=x.device
            )
            idx = None
        # Augmentations
        if training:
            if self.augment is not None and self.augment.prob > 0:
                with torch.inference_mode():
                    x, a = self.augment(x)
                x, a = x.clone(), a.clone()
        # Loss
        loss, logdict_iter = model.loss(x, y=y, a=a, idx=idx, rng=rng)
        # Backward
        if training:
            self.optim.zero_grad(set_to_none=True)
            self.fabric.backward(loss)
            """
            # Fix nan gradient if needed
            # (from https://github.com/NVlabs/edm/blob/main/training/training_loop.py)
            for param in model.parameters():
                if param.grad is not None:
                    torch.nan_to_num(
                        param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                    )
            #"""
            self.optim.step()
        # Log more info
        with torch.inference_mode():
            logdict_iter["l_wmag"] = pytorch_utils.compute_weights_average(model)
        logdict.append(logdict_iter)
        return logdict

    ###########################################################################

    def _print_losses(self, logdict, frequency=0.65, force=False):
        if self.fabric.is_global_zero:
            new_time = print_utils.get_time_base(frequency)
            if force or new_time != self.current_time:
                losses = logdict.get("l_main")
                self.fabric.print(
                    f" [L*={losses[-1]:.3f}, L={losses.mean():.3f}] ", end="\r"
                )
                self.current_time = new_time

    @torch.inference_mode()
    def _checkpointing(self, epoch, cost, is_best=False):
        state = pytorch_utils.create_state(
            model=self.model,
            ema=self.ema,
            optim=self.optim,
            sched=self.sched,
            conf=self.all_conf,
            epoch=epoch + 1,
            cost=cost,
        )
        fn_last, fn_best, fn_epoch = get_checkpoint_names(self.all_conf.path.logs)
        self.fabric.save(fn_last, state)
        if is_best:
            self.fabric.save(fn_best, state)
        if (
            self.conf.checkpoint_mult_epoch is not None
            and (epoch + 1) % self.conf.checkpoint_mult_epoch == 0
        ):
            sepoch = str(epoch + 1).zfill(len(str(self.conf.num_epochs)))
            fn = fn_epoch.replace("$epoch$", "epoch-" + sepoch)
            self.fabric.save(fn, state)  # periodic


###############################################################################
