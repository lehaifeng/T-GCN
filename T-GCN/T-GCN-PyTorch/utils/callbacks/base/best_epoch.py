import copy
import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.callbacks import Callback


class BestEpochCallback(Callback):
    TORCH_INF = torch_inf = torch.tensor(np.Inf)
    MODE_DICT = {
        "min": (torch_inf, "min"),
        "max": (-torch_inf, "max"),
    }
    MONITOR_OP_DICT = {"min": torch.lt, "max": torch.gt}

    def __init__(self, monitor="", mode="min"):
        super(BestEpochCallback, self).__init__()
        self.monitor = monitor
        self.__init_monitor_mode(monitor, mode)
        self.best_epoch = 0

    def __init_monitor_mode(self, monitor, mode):
        if mode not in self.MODE_DICT and mode != "auto":
            rank_zero_warn(
                f"PrintBestEpochMetrics mode {mode} is unknown, fallback to auto mode",
                RuntimeWarning,
            )
            mode = "auto"
        if mode == "auto":
            rank_zero_warn(
                "mode='auto' is deprecated in v1.1 and will be removed in v1.3."
                " Default value for mode with be 'min' in v1.3.",
                DeprecationWarning,
            )
            self.MODE_DICT["auto"] = (
                (-self.TORCH_INF, "max")
                if monitor is not None and ("acc" in monitor or monitor.startswith("fmeasure"))
                else (self.TORCH_INF, "min")
            )
        self.best_value, self.mode = self.MODE_DICT[mode]

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if (trainer.current_epoch + 1) % trainer.check_val_every_n_epoch != 0:
            return
        monitor_op = self.MONITOR_OP_DICT[self.mode]
        metrics_dict = copy.copy(trainer.callback_metrics)
        monitor_value = metrics_dict.get(self.monitor, self.best_value)
        if monitor_op(monitor_value.type_as(self.best_value), self.best_value):
            self.best_value = monitor_value
            self.best_epoch = trainer.current_epoch
