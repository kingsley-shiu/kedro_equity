"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""

from kedro.pipeline import Pipeline, node

from . import nodes


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=nodes.get_nasdaq_symbols,
                inputs=None,
                outputs="nasdaq_symbol",
                name="nasdaq_symbol",
                tags="gen",
            ),
            node(
                func=nodes.get_nasdaq_daily_price,
                inputs="nasdaq_symbol",
                outputs="nasdaq_daily_price",
                name="nasdaq_daily_price",
                tags="gen",
            ),
            node(
                func=nodes.get_nasdaq_company_info,
                inputs=["nasdaq_symbol", "nasdaq_company_info_loaded"],
                outputs="nasdaq_company_info",
                name="nasdaq_company_info",
                tags="gen",
            ),
            node(
                func=nodes.get_nasdaq_share_held,
                inputs=["nasdaq_symbol", "nasdaq_share_held_loaded"],
                outputs="nasdaq_share_held",
                name="nasdaq_share_held",
                tags="gen",
            ),
        ]
    )
