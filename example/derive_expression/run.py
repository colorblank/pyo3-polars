import polars as pl
from datetime import date, datetime, timezone
from expression_lib import language, dist, date_util, panic
import time

df = pl.DataFrame(
    {
        "names": ["Richard", "Alice", "Bob"],
        "moons": ["full", "half", "red"],
        "dates": [date(2023, 1, 1), date(2024, 1, 1), date(2025, 1, 1)],
        "datetime": [datetime.now(tz=timezone.utc)] * 3,
        "dist_a": [[12, 32, 1], [], [1, -2]],
        "dist_b": [[-12, 1], [43], [876, -45, 9]],
        "start_lat": [5.6, -1245.8, 242.224],
        "start_lon": [3.1, -1.1, 128.9],
        "end_lat": [6.6, -1243.8, 240.224],
        "end_lon": [3.9, -2, 130],
        "data_str": ["aa#0.5|bbb#0.3|ccc#0.2", "x#1.0|y#2.0", "p#0.1"],
    }
)

out = df.with_columns(
    pig_latin=language.pig_latinnify("names"),
    pig_latin_cap=language.pig_latinnify("names", capitalize=True),
).with_columns(
    hamming_dist=dist.hamming_distance("names", "pig_latin"),
    jaccard_sim=dist.jaccard_similarity("dist_a", "dist_b"),
    haversine=dist.haversine("start_lat", "start_lon", "end_lat", "end_lon"),
    leap_year=date_util.is_leap_year("dates"),
    new_tz=date_util.change_time_zone("datetime"),
    appended_args=language.append_args(
        "names",
        float_arg=11.234,
        integer_arg=93,
        boolean_arg=False,
        string_arg="example",
    ),
    extracted_and_padded=language.extract_and_pad(
        "data_str", sep1="|", sep2="#", index=0, max_len=5, pad_value="NULL"
    ),
)

print(out)

# Test we can extend the expressions by importing the extension module.

import expression_lib.extension  # noqa: F401

out = df.with_columns(
    pig_latin=pl.col("names").language.pig_latinnify(),
    pig_latin_cap=pl.col("names").language.pig_latinnify(capitalize=True),
).with_columns(
    hamming_dist=pl.col("names").dist.hamming_distance("pig_latin"),
    jaccard_sim=pl.col("dist_a").dist.jaccard_similarity("dist_b"),
    haversine=pl.col("start_lat").dist.haversine("start_lon", "end_lat", "end_lon"),
    leap_year=pl.col("dates").date_util.is_leap_year(),
    new_tz=pl.col("datetime").date_util.change_time_zone(),
    appended_args=pl.col("names").language.append_args(
        float_arg=11.234,
        integer_arg=93,
        boolean_arg=False,
        string_arg="example",
    ),
    extracted_and_padded_expr=pl.col("data_str").language.extract_and_pad(
        sep1="|", sep2="#", index=0, max_len=5, pad_value="NULL"
    ),
)

print(out)

# Python implementation for comparison
def python_extract_and_pad(
    s: str, sep1: str = "|", sep2: str = "#", index: int = 0, max_len: int = 5, pad_value: str = "NULL"
) -> list[str]:
    extracted_elements = []
    if s is not None:
        parts1 = s.split(sep1)
        for part1 in parts1:
            parts2 = part1.split(sep2)
            if index < len(parts2):
                extracted_elements.append(parts2[index])
            else:
                extracted_elements.append(pad_value)

    current_len = len(extracted_elements)
    if current_len < max_len:
        extracted_elements.extend([pad_value] * (max_len - current_len))
    elif current_len > max_len:
        extracted_elements = extracted_elements[:max_len]
    return extracted_elements

# Performance comparison
print("\n--- Performance Comparison ---")
N_ROWS = 1_000_000
test_df = pl.DataFrame({
    "data_str": ["aa#0.5|bbb#0.3|ccc#0.2"] * N_ROWS
})

# Rust UDF performance
start_time = time.perf_counter()
rust_out = test_df.with_columns(
    pl.col("data_str").language.extract_and_pad(
        sep1="|", sep2="#", index=0, max_len=5, pad_value="NULL"
    ).alias("rust_result")
)
end_time = time.perf_counter()
rust_time = end_time - start_time
print(f"Rust UDF execution time: {rust_time:.4f} seconds")

# Python UDF performance (using apply)
start_time = time.perf_counter()
python_out = test_df.with_columns(
    pl.col("data_str").map_elements(
        lambda s: python_extract_and_pad(s, sep1="|", sep2="#", index=0, max_len=5, pad_value="NULL"),
        return_dtype=pl.List(pl.String)
    ).alias("python_result")
)
end_time = time.perf_counter()
python_time = end_time - start_time
print(f"Python UDF (apply) execution time: {python_time:.4f} seconds")

# Tests we can return errors from FFI by passing wrong types.
try:
    out.with_columns(
        appended_args=pl.col("names").language.append_args(
            float_arg=True,
            integer_arg=True,
            boolean_arg=True,
            string_arg="example",
        )
    )
except pl.exceptions.ComputeError as e:
    assert "the plugin failed with message" in str(e)


try:
    out.with_columns(pl.col("names").panic.panic())
except pl.exceptions.ComputeError as e:
    assert "the plugin panicked" in str(e)

print("finished")
