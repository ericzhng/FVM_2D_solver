r"""Module: tests.unit.test_limiters

Description: Unit tests for limiter lookup and the limiter functions (range and values).

Author: Hui Zhang
Date: 2026-11-18
Copyright: (c) 2026 Hui Zhang. All rights reserved.
"""

import pytest

from src.limiters import (
    BARTH_JESPERSEN,
    LIMITER_CODES,
    VENKATAKRISHNAN,
    apply_limiter,
    limiter_code,
)


def test_codes_and_deprecated_alias():
    assert limiter_code("barth_jespersen") == BARTH_JESPERSEN
    assert limiter_code("venkatakrishnan") == VENKATAKRISHNAN
    assert limiter_code("minmod") == BARTH_JESPERSEN
    assert set(LIMITER_CODES) >= {"barth_jespersen", "venkatakrishnan"}


def test_unknown_limiter_names_the_choices():
    with pytest.raises(ValueError, match="barth_jespersen"):
        limiter_code("superbee")


def test_barth_jespersen_is_min_of_one_and_bound_ratio():
    assert apply_limiter(BARTH_JESPERSEN, 0.5, 1.0, 0.0) == pytest.approx(0.5)
    assert apply_limiter(BARTH_JESPERSEN, 2.0, 1.0, 0.0) == 1.0  # within bounds: unlimited
    assert apply_limiter(BARTH_JESPERSEN, -0.5, -2.0, 0.0) == pytest.approx(0.25)


@pytest.mark.parametrize("eps2", [0.0, 1e-3, 1e-1])
@pytest.mark.parametrize("ratio", [1e-3, 0.1, 0.5, 1.0, 3.0, 1e3])
def test_venkatakrishnan_stays_in_unit_interval(ratio, eps2):
    phi = apply_limiter(VENKATAKRISHNAN, 1.0, ratio, eps2)
    assert 0.0 <= phi <= 1.0


def test_venkatakrishnan_large_smoothing_is_nearly_unlimited():
    assert apply_limiter(VENKATAKRISHNAN, 1.0, 1.5, 1e3) > 0.99
