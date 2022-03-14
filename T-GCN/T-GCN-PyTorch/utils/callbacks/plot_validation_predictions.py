import numpy as np
import matplotlib.pyplot as plt
from utils.callbacks.base import BestEpochCallback


class PlotValidationPredictionsCallback(BestEpochCallback):
    def __init__(self, monitor="", mode="min"):
        super(PlotValidationPredictionsCallback, self).__init__(monitor=monitor, mode=mode)
        self.ground_truths = []
        self.predictions = []

    def on_fit_start(self, trainer, pl_module):
        self.ground_truths.clear()
        self.predictions.clear()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if trainer.current_epoch != self.best_epoch:
            return
        self.ground_truths.clear()
        self.predictions.clear()
        predictions, y = outputs
        predictions = predictions.cpu().numpy()
        y = y.cpu().numpy()
        self.ground_truths.append(y[:, 0, :])
        self.predictions.append(predictions[:, 0, :])

    def on_fit_end(self, trainer, pl_module):
        ground_truth = np.concatenate(self.ground_truths, 0)
        predictions = np.concatenate(self.predictions, 0)
        tensorboard = pl_module.logger.experiment
        for node_idx in range(ground_truth.shape[1]):
            plt.clf()
            plt.rcParams["font.family"] = "Times New Roman"
            fig = plt.figure(figsize=(7, 2), dpi=300)
            plt.plot(
                ground_truth[:, node_idx],
                color="dimgray",
                linestyle="-",
                label="Ground truth",
            )
            plt.plot(
                predictions[:, node_idx],
                color="deepskyblue",
                linestyle="-",
                label="Predictions",
            )
            plt.legend(loc="best", fontsize=10)
            plt.xlabel("Time")
            plt.ylabel("Traffic Speed")
            tensorboard.add_figure(
                "Prediction result of node " + str(node_idx),
                fig,
                global_step=len(trainer.train_dataloader) * self.best_epoch,
                close=True,
            )
