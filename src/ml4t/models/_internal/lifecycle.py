"""Atomic fitted-state lifecycle helpers."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, cast

type Method = Callable[..., Any]


def atomic_fit(*attribute_names: str) -> Callable[[Method], Method]:
    def decorate(method: Method) -> Method:
        @wraps(method)
        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            previous = tuple(getattr(self, name) for name in attribute_names)
            try:
                return method(self, *args, **kwargs)
            except BaseException:
                for name, value in zip(attribute_names, previous, strict=True):
                    setattr(self, name, value)
                raise

        return cast(Method, wrapped)

    return decorate
