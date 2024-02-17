import pytest

from neurodesign import msequence


@pytest.mark.parametrize("stimtypeno", [2, 4, 8, 9])
def test_msequence_smoke(stimtypeno):
    ntrials = 100
    order = msequence.Msequence()
    order.GenMseq(mLen=ntrials, stimtypeno=stimtypeno, seed=42)
