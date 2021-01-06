import numpy as np

from . import batcher

def test_streaming_mean():
    m = batcher.StreamingMean()
    values = list(range(10, 20))

    for i, value in enumerate(values):
        m.add(value)
        assert m.value == np.mean(values[:i + 1])