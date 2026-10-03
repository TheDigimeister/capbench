import numpy as np
import pytest

from capbench import traces


@pytest.mark.parametrize("name", list(traces.SOURCES))
def test_trace_window_hits_target_rate(name):
    if not (traces.RAW / traces.SOURCES[name][0]).exists():
        pytest.skip("trace not downloaded")
    t = traces.arrivals(name, 3000, rate=2.5, seed=1)
    assert len(t) == 3000 and (np.diff(t) >= 0).all() and t[0] >= 0
    assert 3000 / t[-1] == pytest.approx(2.5, rel=1e-9)
    assert not np.array_equal(t, traces.arrivals(name, 3000, rate=2.5, seed=2))
