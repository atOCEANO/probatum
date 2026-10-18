"""The API page's Indicators section runs as written after its Quickstart, in one
namespace, as a reader copies them: each strategy it shows runs on the Quickstart's Data
and trades, and the universe it builds gives each instrument its own average, a gap
where ETHUSDT printed nothing and after it the average of the bars ETHUSDT printed.
"""

import numpy as np

from probatum import ta
from test_the_api_page_s_quickstart_and_result_run import PAGE, sections


def test_the_indicators_section_runs_after_the_quickstart():
    blocks = sections("Quickstart", "Indicators")

    assert len(blocks) == 4

    names, results = {}, []
    for block in blocks:
        exec(compile(block, str(PAGE), "exec"), names)
        results.append(names["result"])

    for result in results[1:3]:
        assert type(result.policy).__name__ == "Cross" and len(result.fills) > 0

    close, trend = names["bars"][:, 3], names["trend"]

    assert trend.shape == (1_000, 2)
    assert np.array_equal(trend[:, 0], ta.ema(close, 50), equal_nan=True)
    assert np.isnan(trend[500:510, 1]).all()

    printed = ta.ema(np.delete(close, np.s_[500:510]), 50)

    assert np.array_equal(trend[:500, 1], printed[:500], equal_nan=True)
    assert np.array_equal(trend[510:, 1], printed[500:])
