import pytest
from training import LunaTrainingApp
from conftest import luna_setup as setup 


def test_main(setup):
    app = LunaTrainingApp(sys_argv=[
        f'--data-dir={setup.data_dir}',
        f'--val-stride=10',
    ])
    app.main()





        
