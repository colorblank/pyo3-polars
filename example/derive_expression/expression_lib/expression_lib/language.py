from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from expression_lib._utils import LIB

if TYPE_CHECKING:
    from expression_lib._typing import IntoExprColumn


def pig_latinnify(expr: IntoExprColumn, capitalize: bool = False) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr],
        function_name="pig_latinnify",
        is_elementwise=True,
        kwargs={"capitalize": capitalize},
    )


def hash_and_modulus(expr: IntoExprColumn, dictionary_size: int) -> pl.Expr:
    """
    将输入的字符串列表通过hash映射为数值，然后通过取余数，约束到指定的字典大小。
    Args:
        expr (IntoExprColumn): 输入的字符串列。
        dictionary_size (int): 字典大小，用于取余数。
    Returns:
        pl.Expr: 包含哈希并取模后数值的表达式。
    Examples:
        >>> import polars as pl
        >>> from expression_lib import language
        >>> df = pl.DataFrame({"strings": ["aa", "bbb", "ccc", "null", "null"]})
        >>> df.with_columns(language.hash_and_modulus("strings", dictionary_size=10))
        shape: (5, 2)
        ┌─────────┬──────────────────┐
        │ strings ┆ strings_hashed   │
        │ str     ┆ u32              │
        ╞═════════╪══════════════════╡
        │ aa      ┆ 2                │
        │ bbb     ┆ 9                │
        │ ccc     ┆ 0                │
        │ null    ┆ 5                │
        │ null    ┆ 5                │
        └─────────┴──────────────────┘
    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr],
        function_name="hash_and_modulus",
        is_elementwise=True,
        kwargs={"dictionary_size": dictionary_size},
    )


def append_args(
    expr: IntoExprColumn,
    float_arg: float,
    integer_arg: int,
    string_arg: str,
    boolean_arg: bool,
) -> pl.Expr:
    """
    This example shows how arguments other than `Series` can be used.
    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr],
        kwargs={
            "float_arg": float_arg,
            "integer_arg": integer_arg,
            "string_arg": string_arg,
            "boolean_arg": boolean_arg,
        },
        function_name="append_kwargs",
        is_elementwise=True,
    )


def extract_and_pad(
    expr: IntoExprColumn,
    sep1: str = "|",
    sep2: str = "#",
    index: int = 0,
    max_len: int = 5,
    pad_value: str = "NULL",
) -> pl.Expr:
    """
    Splits a string by sep1, then splits each part by sep2 and extracts the element at 'index'.
    Pads the resulting list to 'max_len' with 'pad_value'.
    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr],
        kwargs={
            "sep1": sep1,
            "sep2": sep2,
            "index": index,
            "max_len": max_len,
            "pad_value": pad_value,
        },
        function_name="extract_and_pad",
        is_elementwise=True,
    )
