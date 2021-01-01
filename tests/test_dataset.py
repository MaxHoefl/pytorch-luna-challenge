import pytest
from conftest import luna_setup
from dataset import LunaDataset


def test_getitem(luna_setup):
    ds = LunaDataset(luna_setup.data_dir)
    assert ds[0][0].shape == (1, 32, 48, 48)
    assert len(ds) == 1

