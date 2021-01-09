import argparse
import sys
import os
import datetime 
import logging as log
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from model import LunaModel
from dataset import LunaDataset
import logging as log
log.basicConfig(
    level=log.INFO,
    format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


DATA_DIR = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data-unversioned')


class LunaTrainingApp(object):
    METRICS_LOSS_IDX = 2
    METRICS_PREDS_IDX = 1
    METRICS_LABELS_IDX = 0

    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        log.info(f'Creating training app: {sys_argv}')
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=1,
            type=int
        )
        parser.add_argument('--data-dir',
            help='Data directory',
            default=DATA_DIR,
            type=str
        )
        parser.add_argument('--batch-size',
            help='Batch size',
            default=32,
            type=int
        )
        parser.add_argument('--epochs',
            help='Number of epochs',
            default=1,
            type=int
        )
        parser.add_argument('--val-stride',
            help='Multiples of which will be included in val dataset',
            default=10,
            type=int
        )
        parser.add_argument('--conv-channels',
            help='Number of output channels for first convolution',
            default=8,
            type=int
        )
        self.cli_args = parser.parse_args(sys_argv)
        if self.cli_args.val_stride < 0:
            self.cli_args.val_stride = None
        self.now = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

    def main(self):
        log.info(f'Starting {type(self).__name__}, {self.cli_args}')
        train_dl = self.init_dataloader(
                mode='train', 
                val_stride=self.cli_args.val_stride)
        val_dl = self.init_dataloader(
                mode='val',
                val_stride=self.cli_args.val_stride)
        train_metrics = self.init_metrics(
            num_epochs=self.cli_args.epochs,
            num_batches=len(train_dl),
            batch_size=train_dl.batch_size
        )
        val_metrics = self.init_metrics(
            num_epochs=self.cli_args.epochs,
            num_batches=len(val_dl),
            batch_size=val_dl.batch_size
        )
        for epoch_idx in range(self.cli_args.epochs):
            log.info(f'Epoch {epoch_idx}')
            train_metrics = self.training_epoch(
                epoch_idx, train_dl, train_metrics
            )
            self.log_epoch_metrics(train_metrics, epoch_idx)

    def init_metrics(self, num_epochs, num_batches, batch_size):
        num_metrics = 3 # label, pred, loss
        return torch.zeros(
            (num_epochs, num_batches, batch_size, num_metrics),
            device=self.device
        )

    def training_epoch(self, epoch_idx, train_dl, metrics=None):
        self.model.train()
        for batch_idx, batch in enumerate(train_dl):
            self.optimizer.zero_grad()
            loss = self.compute_loss(
                epoch_idx,
                batch_idx,
                batch,
                train_dl.batch_size,
                metrics
            )
            loss.backward()
            self.optimizer.step()
        if metrics is not None:
            return metrics.to('cpu')

    def log_epoch_metrics(self, metrics, epoch_idx):
        avg_loss = metrics[epoch_idx, :, :, self.METRICS_LOSS_IDX].mean()
        log.info(f"epoch: {epoch_idx}, loss: {avg_loss}")

    def validation_epoch(self, epoch_idx, val_dl, metrics=None):
        with torch.no_grad():
            self.model.eval()
            for batch_idx, batch in enumerate(val_dl):
                loss.self.compute_loss(
                    epoch_idx,
                    batch_idx,
                    batch,
                    val_dl.batch_size,
                    metrics
                )
            if metrics is not None:
                return metrics.to('cpu')

    def compute_loss(self, epoch_idx, batch_idx, batch, batch_size, metrics=None):
        inputs, labels, series_uids, centers = batch
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        inputs = inputs.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        logits, preds = self.model(inputs)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fn(logits, labels[:,1])
        if metrics is not None:
            metrics[epoch_idx, batch_idx, start_idx:end_idx, 0] = \
                    labels[:, 1].detach()
            metrics[epoch_idx, batch_idx, start_idx:end_idx, 1] = \
                    preds[:, 1].detach() 
            metrics[epoch_idx, batch_idx, start_idx:end_idx, 2] = \
                    loss.detach()
        return loss.mean()

    def init_dataloader(self, mode='train', val_stride=10):
        log.info(f'Creating dataloader - {mode}')
        ds = LunaDataset(
            data_dir=self.cli_args.data_dir,
            val_stride=val_stride,
            is_val=mode.lower().startswith('val')
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
            print(f"BATCH_SIZE: {batch_size}")
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )
        return dl

    def init_model(self):
        model = LunaModel(
            in_channels=LunaDataset.INPUT_CHANNELS,
            conv_channels=self.cli_args.conv_channels,
            depth=LunaDataset.CROP_DEPTH,
            height=LunaDataset.CROP_HEIGHT,
            width=LunaDataset.CROP_WIDTH
        )
        if self.use_cuda:
            log.info(f'Using CUDA; {torch.cuda.device_count()} devices')
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def init_optimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)


if __name__ == '__main__':
    LunaTrainingApp().main()
