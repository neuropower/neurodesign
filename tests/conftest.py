import pytest

from neurodesign import experiment


@pytest.fixture
def exp():

    return experiment(
        TR=2,
        n_trials=20,
        P=[0.3, 0.3, 0.4],
        C=[[1, -1, 0], [0, 1, -1]],
        n_stimuli=3,
        rho=0.3,
        stim_duration=1,
        t_pre=0.5,
        t_post=2,
        ITImodel="exponential",
        ITImin=2,
        ITImax=4,
        ITImean=2.1,
    )
