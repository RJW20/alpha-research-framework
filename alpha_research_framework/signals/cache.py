from typing import TYPE_CHECKING, TypeAlias

import alpha_research_framework.cross_section as xs

if TYPE_CHECKING:
    from .signal import Signal

Cache: TypeAlias = dict[type["Signal"], xs.Array]
