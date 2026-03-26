from typing import TYPE_CHECKING, TypeAlias

import alpha_research_framework.cross_section as xs

if TYPE_CHECKING:
    from .factor import Factor

FactorCache: TypeAlias = dict[type["Factor"], xs.Array]
