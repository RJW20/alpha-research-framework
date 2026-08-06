from typing import Iterable

import alpha_research_framework.market_data as md

from .cache import Cache
from .series import Series


def build(
    series: Iterable[type[Series]],
    *,
    market_data: md.MarketData,
    allocator: md.Allocator,
) -> dict[type[Series], md.Array]:
    """
    Return a dict mapping `type[Series]` to an `md.Array` containing its
    relevant data for every requested `Series` in `series`.
    
    Temporary `md.Array` objects may be allocated and released during execution.
    """

    cache = Cache()
    built = {s: s(market_data, cache, allocator) for s in series}

    for s in cache.keys() - series:
        allocator.release(cache[s])

    return built
