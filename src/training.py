import argparse
import sys
import datetime 
import logging as log
import torch
from torch import nn
from torch.optim import SGD
from model import LunaModel


class LunaTrainingApp(object):
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int
        )
        self.cli_args = parser.parse_args(sys_argv)
        self.now = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

    def main(self):
        log.info(f'Starting {type(self).__name__}, {self.cli_args}')

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
