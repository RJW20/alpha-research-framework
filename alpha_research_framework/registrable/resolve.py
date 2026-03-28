from typing import TypeVar

from .registrable import Registrable

T = TypeVar('T', bound=Registrable)
def resolve(
    registrable: str | type[T],
    registrable_subtype: type[T],
) -> type[T]:
    """
    Return the subtype of `registrable_subtype` referenced by `registrable`.

    If `registrable` is of type `str` the corresponding type will be returned.
    If `registrable` is of type `type` it will validated as a subtype of
    `registrable_subtype` before being returned back.
    """
    
    if isinstance(registrable, str):
        return registrable_subtype.from_id(registrable)
    if not issubclass(registrable, registrable_subtype):
        raise TypeError(
            f"Type {registrable} must be a subclass of "
            f"{registrable_subtype.__name__}"
        )
    return registrable
