import copy
import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from utils.callbacks.base import BestEpochCallback
from pl_bolts.callbacks.printing import dicts_to_table


class PrintBestEpochMetricsCallback(BestEpochCallback):
    def __init__(self, monitor='', mode='min'):
        super(PrintBestEpochMetricsCallback, self).__init__(monitor=monitor, mode=mode)
        self.metrics_dict = {}

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if (trainer.current_epoch + 1) % trainer.check_val_every_n_epoch != 0 or \
            trainer.current_epoch != self.best_epoch:
            return
        metrics_dict = copy.copy(trainer.callback_metrics)
        self.metrics_dict = metrics_dict

    def on_fit_end(self, trainer, pl_module):
        rank_zero_info(dicts_to_table([self.metrics_dict]))
