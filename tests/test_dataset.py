import pytest
from conftest import luna_setup
from dataset import LunaDataset


def test_dateset(luna_setup):
    ds = LunaDataset(luna_setup.data_dir)
    assert ds[0][0].shape == (
            LunaDataset.INPUT_CHANNELS, 
            LunaDataset.CROP_DEPTH, 
            LunaDataset.CROP_HEIGHT, 
            LunaDataset.CROP_WIDTH)
    assert len(ds) == 1


def test_fullds(luna_setup):
    ds = LunaDataset(
        data_dir=luna_setup.data_dir,
        val_stride=None,
        is_val=False
    )
    assert len(ds) == 1


