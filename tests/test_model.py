import pytest
from model import LunaBlock, LunaModel
import torch


class Setup(object):
    def __init__(self):
        self.batch_size = 2
        self.channels = 1
        self.depth = 50
        self.height = 100
        self.width = 100
        self.sample_batch = torch.rand((
            self.batch_size, self.channels, self.depth, self.height, self.width
        ))


@pytest.fixture
def setup():
    return Setup()


def test_lunablock_forward(setup):
    block = LunaBlock(in_channels=setup.channels, conv_channels=1)
    out = block.forward(setup.sample_batch)
    assert out.shape == (setup.batch_size, 
                         setup.channels, 
                         setup.depth // 2, 
                         setup.height // 2, 
                         setup.width // 2)


def test_lunamodel_forward(setup):
    model = LunaModel(
            in_channels=setup.channels, 
            conv_channels=1,
            depth=setup.depth,
            height=setup.height,
            width=setup.width
    )
    logits, pred = model.forward(setup.sample_batch)
    assert pred.shape == (setup.batch_size, 2)

