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


DATA_DIR = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data-unversioned')


class LunaTrainingApp(object):
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
        for epoch_idx in range(1, self.cli_args.epochs + 1):
            log.info(f'Epoch {epoch_idx}')
            self.training_epoch(epoch_idx, train_dl)

    def training_epoch(self, epoch_idx, train_dl):
        self.model.train()
        for batch_idx, batch in enumerate(train_dl):
            self.optimizer.zero_grad()
            loss = self.compute_loss(
                batch_idx,
                batch,
                train_dl.batch_size
            )
            loss.backward()
            self.optimizer.step()

    def compute_loss(self, batch_idx, batch, batch_size):
        inputs, labels, series_uids, centers = batch
        inputs = inputs.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        logits, preds = self.model(inputs)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fn(logits, labels[:,1])
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
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )
        return dl

    def init_model(self):
        model = LunaModel()
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
