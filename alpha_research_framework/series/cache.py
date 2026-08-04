from typing import TYPE_CHECKING, TypeAlias

import alpha_research_framework.market_data as md

if TYPE_CHECKING:
    from .series import Series

Cache: TypeAlias = dict[type["Series"], md.Array]
