import pytest

from sentinel_rosetta.infrastructure.opencc_adapter import OpenCcAdapter


@pytest.fixture
def adapter() -> OpenCcAdapter:
    return OpenCcAdapter()


def test_to_simplified_converts_traditional_characters(
    adapter: OpenCcAdapter,
) -> None:
    # Act / Assert
    assert adapter.to_simplified("步兵戰車") == "步兵战车"


def test_to_simplified_leaves_already_simplified_text_unchanged(
    adapter: OpenCcAdapter,
) -> None:
    # Act / Assert
    assert adapter.to_simplified("步兵战车") == "步兵战车"


def test_to_traditional_converts_simplified_characters(
    adapter: OpenCcAdapter,
) -> None:
    # Act / Assert
    assert adapter.to_traditional("猫很快") == "貓很快"


def test_to_traditional_leaves_already_traditional_text_unchanged(
    adapter: OpenCcAdapter,
) -> None:
    # Act / Assert
    assert adapter.to_traditional("步兵戰車") == "步兵戰車"


@pytest.mark.parametrize(
    "method_name",
    ["to_simplified", "to_traditional"],
    ids=["to-simplified", "to-traditional"],
)
def test_script_conversion_empty_string_returns_empty_string(
    adapter: OpenCcAdapter, method_name: str
) -> None:
    # Act
    converted = getattr(adapter, method_name)("")

    # Assert
    assert converted == ""
