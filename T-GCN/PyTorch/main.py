import argparse
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import models
import tasks
import utils.callbacks
import utils.data
import utils.misc


DATA_PATHS = {
    'shenzhen': {
        'feat': 'data/sz_speed.csv',
        'adj': 'data/sz_adj.csv'
    },
    'losloop': {
        'feat': 'data/los_speed.csv',
        'adj': 'data/los_adj.csv'
    }
}


def get_model(args, dm):
    model = None
    if args.model_name == 'GCN':
        model = models.GCN(adj=dm.adj, input_dim=args.seq_len, output_dim=args.hidden_dim)
    if args.model_name == 'GRU':
        model = models.GRU(input_dim=dm.adj.shape[0], hidden_dim=args.hidden_dim)
    if args.model_name == 'TGCN':
        model = models.TGCN(adj=dm.adj, hidden_dim=args.hidden_dim, loss=args.loss)
    return model


def get_task(args, model, dm):
    task = getattr(tasks, args.settings.capitalize() + 'ForecastTask')(model=model, 
                                                                       feat_max_val=dm.feat_max_val,
                                                                       **vars(args))
    return task


def get_callbacks(args):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='train_loss')
    print_best_epoch_metrics_callback = utils.callbacks.PrintBestEpochMetricsCallback(monitor='train_loss')
    plot_validation_predictions_callback = utils.callbacks.PlotValidationPredictionsCallback(monitor='train_loss')
    callbacks = [checkpoint_callback, print_best_epoch_metrics_callback, plot_validation_predictions_callback]
    return callbacks


def main_supervised(args):
    dm = utils.data.SpatioTemporalCSVDataModule(feat_path=DATA_PATHS[args.data]['feat'],
                                                adj_path=DATA_PATHS[args.data]['adj'],
                                                **vars(args))
    model = get_model(args, dm)
    task = get_task(args, model, dm)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=get_callbacks(args))
    trainer.fit(task, dm)


def main(args):
    rank_zero_info(vars(args))
    globals()['main_' + args.settings](args)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument('--data', type=str, help='The name of the dataset',
                        choices=('shenzhen', 'losloop'), default='losloop')
    parser.add_argument('--model_name', type=str, help='The name of the model for spatiotemporal prediction', 
                        choices=('GCN', 'GRU', 'TGCN'), default='GCN')
    parser.add_argument('--settings', type=str, help='The type of settings, e.g. supervised learning',
                        choices=('supervised', ), default='supervised')

    temp_args, _ = parser.parse_known_args()

    parser = getattr(utils.data, temp_args.settings.capitalize() + 'DataModule').add_data_specific_arguments(parser)
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(tasks, temp_args.settings.capitalize() + 'ForecastTask').add_task_specific_arguments(parser)

    args = parser.parse_args()
    utils.misc.format_logger()

    main(args)
