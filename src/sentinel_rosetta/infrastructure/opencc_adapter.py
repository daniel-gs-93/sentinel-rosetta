"""Resource-aware: converts between Chinese scripts with OpenCC."""

from opencc import OpenCC


class OpenCcAdapter:
    def __init__(self) -> None:
        self._t2s = OpenCC("t2s")
        self._s2twp = OpenCC("s2twp")

    def to_traditional(self, text: str) -> str:
        return self._s2twp.convert(text)  # type: ignore[no-any-return]

    def to_simplified(self, text: str) -> str:
        return self._t2s.convert(text)  # type: ignore[no-any-return]
