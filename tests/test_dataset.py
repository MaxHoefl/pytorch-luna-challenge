import pytest
from conftest import luna_setup
from dataset import LunaDataset


def test_dateset(luna_setup):
    ds = LunaDataset(luna_setup.data_dir)
    assert ds[0][0].shape == (1, 32, 48, 48)
    assert len(ds) == 1


def test_fullds(luna_setup):
    ds = LunaDataset(
        data_dir=luna_setup.data_dir,
        val_stride=None,
        is_val=False
    )
    assert len(ds) == 1


