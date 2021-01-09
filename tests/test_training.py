import pytest
from training import LunaTrainingApp
from conftest import luna_setup as setup 


def test_main(setup):
    app = LunaTrainingApp(sys_argv=[
        f'--data-dir={setup.data_dir}',
        f'--val-stride=10',
        f'--batch-size=1'
    ])
    app.main()


def test_training_epoch(setup):
    app = LunaTrainingApp(sys_argv=[
        f'--data-dir={setup.data_dir}',
        f'--val-stride=10',
        f'--batch-size=1'
    ])
    train_dl = app.init_dataloader(
        mode='train', 
        val_stride=app.cli_args.val_stride)
    train_metrics = app.init_metrics(
        num_epochs=1,
        num_batches=len(train_dl),
        batch_size=train_dl.batch_size
    )
    print(train_metrics.shape)
    metrics = app.training_epoch(epoch_idx=0, train_dl=train_dl, metrics=train_metrics)








        
