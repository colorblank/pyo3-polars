# Polars Rust UDF 开发指南

## 1. 引言

Polars 是一个高性能的 DataFrame 库，它使用 Rust 编写，并提供了 Python 绑定。为了进一步扩展 Polars 的功能并满足特定需求，用户可以编写自定义的 Rust 用户定义函数 (UDF)。这些 UDF 可以直接在 Polars 的表达式引擎中运行，从而实现高性能的数据处理。

本指南将详细介绍如何使用 `pyo3-polars` 库在 Rust 中实现 Polars UDF，并将其绑定到 Python API，以便在 Polars DataFrame 中使用。我们将通过分析 `example/derive_expression/` 示例项目来深入理解整个开发流程。

## 2. 项目结构

`example/derive_expression/` 目录是一个典型的 `pyo3-polars` 项目结构，用于演示如何构建和使用 Rust UDF。其主要组成部分如下：

```
example/derive_expression/
├── Makefile
├── expression_lib/
│   ├── Cargo.toml
│   ├── pyproject.toml
│   ├── expression_lib/
│   │   ├── __init__.py
│   │   ├── _typing.py
│   │   ├── _utils.py
│   │   ├── date_util.py
│   │   ├── dist.py
│   │   ├── extension.py
│   │   └── language.py
│   └── src/
│       ├── distances.rs
│       ├── expressions.rs
│       └── lib.rs
├── requirements.txt
├── run.py
└── venv/
```

*   **`Makefile`**: 包含用于设置虚拟环境、安装 Rust 模块（通过 `maturin`）和运行示例的便捷命令。
*   **`requirements.txt`**: 列出了 Python 项目的依赖，主要包括 `maturin`（用于 Rust 和 Python 绑定）和 `polars`。
*   **`run.py`**: 示例 Python 脚本，演示了如何导入和使用 Rust 实现的 Polars UDF。
*   **`venv/`**: Python 虚拟环境目录。
*   **`expression_lib/`**: 包含 Rust 源代码和 Python 绑定配置的核心目录。
    *   **`Cargo.toml`**: Rust 项目的清单文件，定义了 Rust 依赖和项目元数据。
    *   **`pyproject.toml`**: Python 项目的构建系统配置，`maturin` 会读取此文件来构建 Python 包。
    *   **`expression_lib/` (Python 包)**: 这是一个 Python 包，其中包含用于将 Rust UDF 暴露给 Python 的辅助文件。
        *   `__init__.py`: Python 包的初始化文件。
        *   `extension.py`: 包含将 Rust UDF 作为 `pl.Expr` 方法调用的扩展逻辑。
        *   `language.py`, `dist.py`, `date_util.py`, `panic.py`: 这些文件定义了 Python 端的函数，它们是 Rust UDF 的 Python 封装。
    *   **`src/` (Rust 源代码)**: 包含实际的 Rust UDF 实现。
        *   `lib.rs`: Rust 库的入口点，通常用于导入其他模块和设置全局分配器。
        *   `expressions.rs`: 包含多个使用 `#[polars_expr]` 宏定义的 UDF，例如字符串处理、日期处理和参数传递示例。
        *   `distances.rs`: 包含一些辅助函数，实现了具体的距离计算逻辑（如海明距离、Jaccard 相似度、Haversine 距离），这些函数被 `expressions.rs` 中的 UDF 调用。

## 3. Rust UDF 实现

Polars Rust UDF 的核心在于使用 `pyo3-polars` 提供的宏和类型来定义函数，使其能够与 Polars 的表达式引擎无缝集成。

### 3.1 依赖管理

在 `expression_lib/Cargo.toml` 文件中，您需要添加 `polars` 和 `pyo3-polars` 作为依赖。`pyo3-polars` 提供了 `polars_expr` 宏以及与 Polars 数据结构交互所需的类型。

```toml
[dependencies]
polars = { version = "0.39", features = ["derive", "list", "strings", "temporal", "dtype-datetime", "dtype-date", "timezones"] }
pyo3-polars = { version = "0.10", features = ["derive"] }
# 其他可能的依赖，例如用于并行计算的 rayon
rayon = { version = "1.8", optional = true }

[features]
# 如果您的 UDF 需要并行处理，可以启用此功能
parallel = ["rayon"]
```

在 `expression_lib/src/lib.rs` 中，您通常会设置 `PolarsAllocator` 作为全局内存分配器，以优化 Polars 的内存使用。

```rust
use pyo3_polars::PolarsAllocator;

mod distances;
mod expressions;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();
```

### 3.2 `#[polars_expr]` 宏

`#[polars_expr]` 是 `pyo3-polars` 提供的核心宏，用于将 Rust 函数标记为 Polars UDF。它会自动生成必要的绑定代码，以便 Polars 能够调用您的 Rust 函数。

**函数签名**:
被 `#[polars_expr]` 标记的函数必须遵循特定的签名模式：

```rust
fn my_udf(inputs: &[Series], kwargs: MyKwargs) -> PolarsResult<Series>
```

*   `inputs: &[Series]`: 这是一个切片，包含了所有输入列的数据。每个 `Series` 代表一个 Polars 列。您可以通过索引（例如 `inputs[0]`）访问它们。
*   `kwargs: MyKwargs` (可选): 如果您的 UDF 需要接收 Python 端的关键字参数，您可以在这里定义一个自定义结构体。这个结构体必须实现 `serde::Deserialize` 特性。
*   `PolarsResult<Series>`: 函数必须返回一个 `PolarsResult<Series>`。`PolarsResult` 是一个 `Result` 类型，用于错误处理。`Series` 是 Polars 的核心数据结构，代表一个列。

**示例 (`expressions.rs` 中的 `pig_latinnify`)**:

```rust
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::fmt::Write;

#[derive(Deserialize)]
struct PigLatinKwargs {
    capitalize: bool,
}

// 辅助函数，实现实际的业务逻辑
fn pig_latin_str(value: &str, capitalize: bool, output: &mut String) {
    if let Some(first_char) = value.chars().next() {
        if capitalize {
            for c in value.chars().skip(1).map(|char| char.to_uppercase()) {
                write!(output, "{c}").unwrap()
            }
            write!(output, "AY").unwrap()
        } else {
            let offset = first_char.len_utf8();
            write!(output, "{}{}ay", &value[offset..], first_char).unwrap()
        }
    }
}

#[polars_expr(output_type=String)]
fn pig_latinnify(inputs: &[Series], kwargs: PigLatinKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?; // 获取第一个输入列的字符串数据
    let out: StringChunked = ca.apply_into_string_amortized(|value, output| {
        pig_latin_str(value, kwargs.capitalize, output)
    });
    Ok(out.into_series()) // 将结果转换为 Series 并返回
}

### 3.3 输出类型推断

`#[polars_expr]` 宏允许您指定 UDF 的输出数据类型。这可以通过两种方式实现：

*   **`output_type`**: 当输出类型是固定且已知时，可以直接指定。例如，`#[polars_expr(output_type=String)]` 表示函数将返回 `String` 类型的 `Series`。

    ```rust
    #[polars_expr(output_type=String)]
    fn pig_latinnify(inputs: &[Series], kwargs: PigLatinKwargs) -> PolarsResult<Series> {
        // ...
    }
    ```

*   **`output_type_func`**: 当输出类型依赖于输入类型或运行时逻辑时，可以提供一个函数来动态推断输出类型。这个函数接收 `&[Field]` 作为输入，并返回 `PolarsResult<Field>`。`Field` 包含了列名和数据类型。

    **示例 (`expressions.rs` 中的 `haversine`)**:
    `haversine` 函数的输出类型（`Float32` 或 `Float64`）取决于输入列的浮点类型。

    ```rust
    use polars::prelude::*;
    use polars_plan::dsl::FieldsMapper;
    use pyo3_polars::derive::polars_expr;

    fn haversine_output(input_fields: &[Field]) -> PolarsResult<Field> {
        FieldsMapper::new(input_fields).map_to_float_dtype()
    }

    #[polars_expr(output_type_func=haversine_output)]
    fn haversine(inputs: &[Series]) -> PolarsResult<Series> {
        let out = match inputs[0].dtype() {
            DataType::Float32 => {
                let start_lat = inputs[0].f32().unwrap();
                let start_long = inputs[1].f32().unwrap();
                let end_lat = inputs[2].f32().unwrap();
                let end_long = inputs[3].f32().unwrap();
                crate::distances::naive_haversine(start_lat, start_long, end_lat, end_long)?
                    .into_series()
            }
            DataType::Float64 => {
                let start_lat = inputs[0].f64().unwrap();
                let start_long = inputs[1].f64().unwrap();
                let end_lat = inputs[2].f64().unwrap();
                let end_long = inputs[3].f64().unwrap();
                crate::distances::naive_haversine(start_lat, start_long, end_lat, end_long)?
                    .into_series()
            }
            _ => unimplemented!(),
        };
        Ok(out)
    }
    ```
    `FieldsMapper::new(input_fields).map_to_float_dtype()` 会根据输入字段的类型自动映射到合适的浮点类型。

### 3.4 数据类型处理

在 Rust UDF 中，您需要从 `Series` 中提取具体的数据类型进行处理，并将结果重新封装为 `Series`。

*   **提取数据**: `Series` 提供了多种方法来获取其底层数据，例如：
    *   `.str()?`: 获取 `StringChunked` (字符串类型)。
    *   `.list()?`: 获取 `ListChunked` (列表类型)。
    *   `.f32()?`, `.f64()?`: 获取 `Float32Chunked` 或 `Float64Chunked` (浮点类型)。
    *   `.date()?`: 获取 `DateChunked` (日期类型)。
    *   `.datetime()?`: 获取 `DatetimeChunked` (日期时间类型)。
    *   `.u32()?`, `.i64()?` 等：获取其他整数类型。

    这些方法通常返回 `PolarsResult<ChunkedArray<T>>`，您可以使用 `?` 运算符进行错误传播。

*   **处理数据**: 获取到 `ChunkedArray` 后，您可以使用其提供的迭代器（如 `.iter()`、`.as_date_iter()`）或 `apply` 方法来逐元素处理数据。对于字符串，`apply_into_string_amortized` 是一个高效的选择，它允许您直接写入一个可变字符串缓冲区。

*   **构建 `Series`**: 处理完成后，将结果 `ChunkedArray` 通过 `.into_series()` 方法转换为 `Series` 并返回。

    **示例 (`expressions.rs` 中的 `is_leap_year`)**:

    ```rust
    use polars::prelude::*;
    use pyo3_polars::derive::polars_expr;

    #[polars_expr(output_type=Boolean)]
    fn is_leap_year(input: &[Series]) -> PolarsResult<Series> {
        let input = &input[0];
        let ca = input.date()?; // 获取日期类型数据

        let out: BooleanChunked = ca
            .as_date_iter() // 迭代日期
            .map(|opt_dt| opt_dt.map(|dt| dt.leap_year())) // 判断是否闰年
            .collect_ca(ca.name().clone()); // 收集结果

        Ok(out.into_series()) // 转换为 Series 并返回
    }
    ```

### 3.5 自定义参数 (Kwargs)

如果您的 UDF 需要接收 Python 端的关键字参数，您可以在 `#[polars_expr]` 宏标记的函数签名中添加一个参数，其类型是一个自定义的结构体。这个结构体必须派生 `serde::Deserialize`，以便 `pyo3-polars` 能够从 Python 传递的字典中反序列化参数。

**示例 (`expressions.rs` 中的 `append_kwargs`)**:

```rust
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::fmt::Write;

#[derive(Deserialize)]
pub struct MyKwargs {
    float_arg: f64,
    integer_arg: i64,
    string_arg: String,
    boolean_arg: bool,
}

#[polars_expr(output_type=String)]
fn append_kwargs(input: &[Series], kwargs: MyKwargs) -> PolarsResult<Series> {
    let input = &input[0];
    let input = input.cast(&DataType::String)?; // 确保输入是字符串类型
    let ca = input.str().unwrap();

    Ok(ca
        .apply_into_string_amortized(|val, buf| {
            write!(
                buf,
                "{}-{}-{}-{}-{}",
                val, kwargs.float_arg, kwargs.integer_arg, kwargs.string_arg, kwargs.boolean_arg
            )
            .unwrap()
        })
        .into_series())
}
```

### 3.6 并行计算

`pyo3-polars` 允许您在 UDF 中利用 Polars 的线程池进行并行计算，从而提高处理大型数据集的性能。您可以通过在函数签名中添加 `context: CallerContext` 参数来访问调用上下文，并检查是否允许并行执行。

**示例 (`expressions.rs` 中的 `pig_latinnify_with_paralellism`)**:

```rust
use polars::prelude::*;
use pyo3_polars::derive::{polars_expr, CallerContext};
use pyo3_polars::export::polars_core::POOL;
use serde::Deserialize;
use rayon::prelude::*; // 需要在 Cargo.toml 中启用 "parallel" feature

#[derive(Deserialize)]
struct PigLatinKwargs {
    capitalize: bool,
}

// 辅助函数，实现实际的业务逻辑
fn pig_latin_str(value: &str, capitalize: bool, output: &mut String) {
    // ... (同上)
}

// 辅助函数，用于分割数据块
fn split_offsets(len: usize, n: usize) -> Vec<(usize, usize)> {
    if n == 1 {
        vec![(0, len)]
    } else {
        let chunk_size = len / n;

        (0..n)
            .map(|partition| {
                let offset = partition * chunk_size;
                let len = if partition == (n - 1) {
                    len - offset
                } else {
                    chunk_size
                };
                (partition * chunk_size, len)
            })
            .collect()
    }
}

#[polars_expr(output_type=String)]
fn pig_latinnify_with_paralellism(
    inputs: &[Series],
    context: CallerContext, // 引入 CallerContext
    kwargs: PigLatinKwargs,
) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;

    if context.parallel() {
        // 如果上下文允许并行，则直接使用 apply_into_string_amortized，Polars 会处理并行
        let out: StringChunked = ca.apply_into_string_amortized(|value, output| {
            pig_latin_str(value, kwargs.capitalize, output)
        });
        Ok(out.into_series())
    } else {
        // 否则，手动使用 Rayon 线程池进行并行处理
        POOL.install(|| {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(ca.len(), n_threads);

            let chunks: Vec<_> = splits
                .into_par_iter() // 使用 Rayon 的并行迭代器
                .map(|(offset, len)| {
                    let sliced = ca.slice(offset as i64, len);
                    let out = sliced.apply_into_string_amortized(|value, output| {
                        pig_latin_str(value, kwargs.capitalize, output)
                    });
                    out.downcast_iter().cloned().collect::<Vec<_>>()
                })
                .collect();

            Ok(
                StringChunked::from_chunk_iter(ca.name().clone(), chunks.into_iter().flatten())
                    .into_series(),
            )
        })
    }
}
```
在 `Cargo.toml` 中，您需要为 `rayon` 库启用 `parallel` 功能，并将其设置为可选功能，以便在不需要并行时可以禁用。

### 3.7 错误处理

Rust UDF 应该通过返回 `PolarsResult<Series>` 来处理错误。`PolarsResult` 是 `Result<T, PolarsError>` 的别名。当发生错误时，您可以使用 `polars_bail!` 宏来方便地返回一个 `PolarsError`。

**示例 (`expressions.rs` 中的 `change_time_zone` 辅助函数 `convert_timezone`)**:

```rust
use polars::prelude::*;
use polars_plan::dsl::FieldsMapper;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct TimeZone {
    tz: String,
}

fn convert_timezone(input_fields: &[Field], kwargs: TimeZone) -> PolarsResult<Field> {
    FieldsMapper::new(input_fields).try_map_dtype(|dtype| match dtype {
        DataType::Datetime(tu, _) => Ok(DataType::Datetime(
            *tu,
            datatypes::TimeZone::opt_try_new(Some(kwargs.tz))?,
        )),
        _ => polars_bail!(ComputeError: "expected datetime"), // 使用 polars_bail! 返回错误
    })
}

#[polars_expr(output_type_func_with_kwargs=convert_timezone)]
fn change_time_zone(input: &[Series], kwargs: TimeZone) -> PolarsResult<Series> {
    let input = &input[0];
    let ca = input.datetime()?;

    let mut out = ca.clone();

    let Some(timezone) = datatypes::TimeZone::opt_try_new(Some(kwargs.tz))? else {
        polars_bail!(ComputeError: "expected timezone") // 再次使用 polars_bail!
    };

    out.set_time_zone(timezone)?;
    Ok(out.into_series())
}
```
当 Rust UDF 返回 `PolarsError` 时，Python 端会捕获到一个 `polars.exceptions.ComputeError`。

### 3.8 Panic 处理

如果 Rust UDF 发生 panic（例如，通过 `panic!` 宏或 `todo!()`），`pyo3-polars` 会捕获这个 panic 并将其转换为 Python 端的 `polars.exceptions.ComputeError`，其中包含 panic 的信息。

**示例 (`expressions.rs` 中的 `panic` 函数)**:

```rust
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type=Boolean)]
fn panic(_input: &[Series]) -> PolarsResult<Series> {
    todo!() // 这将导致 panic
}
```
在 Python 端调用此 UDF 时，会抛出 `ComputeError`：

```python
try:
    out.with_columns(pl.col("names").panic.panic())
except pl.exceptions.ComputeError as e:
    assert "the plugin panicked" in str(e)
```

## 4. Python API 绑定

将 Rust UDF 暴露给 Python 需要使用 `maturin` 工具来构建和安装 Rust 模块。一旦安装完成，您就可以在 Python 代码中像使用普通 Python 模块一样导入和调用您的 UDF。

### 4.1 `maturin develop`

`maturin` 是一个用于 Rust 和 Python 之间绑定的工具，它能够将 Rust 库编译为 Python 模块。在开发过程中，`maturin develop` 命令非常有用，它会在开发模式下编译您的 Rust 代码，并将其安装到当前的 Python 环境中，而无需每次更改都重新构建发布版本。

在 `example/derive_expression/` 目录下的 `Makefile` 中，`install` 目标就使用了 `maturin develop`：

```makefile
install: venv
	unset CONDA_PREFIX && \
	source venv/bin/activate && maturin develop -m expression_lib/Cargo.toml
```

这个命令会：
1.  激活 Python 虚拟环境。
2.  使用 `maturin develop` 命令编译 `expression_lib/Cargo.toml` 定义的 Rust 项目。
3.  将编译后的 Rust 库作为 Python 模块安装到虚拟环境中。

### 4.2 导入 UDF

一旦 Rust 模块通过 `maturin` 安装成功，您就可以在 Python 代码中像导入任何其他 Python 模块一样导入您的 UDF。

在 `run.py` 中，您可以看到以下导入语句：

```python
import polars as pl
from expression_lib import language, dist, date_util, panic
```

这里，`expression_lib` 是您的 Rust 模块的 Python 包名，`language`、`dist`、`date_util` 和 `panic` 是在 Rust 中定义的 UDF 组（通过 `lib.rs` 中的 `mod` 声明和 `pyo3-polars` 的内部机制）。

### 4.3 调用 UDF

`pyo3-polars` 提供了两种主要的 UDF 调用方式：

#### 4.3.1 直接作为函数调用

您可以直接将 UDF 作为导入的模块中的函数来调用，并将 Polars 列名（字符串）作为输入。

**示例 (`run.py` 中的第一种调用方式)**:

```python
import polars as pl
from expression_lib import language, dist, date_util, panic

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
)
print(out)
```
在这种方式中，您直接将列名字符串传递给 UDF。Polars 会在内部将这些列名解析为 `Series` 对象，并传递给 Rust UDF。

#### 4.3.2 作为 `pl.Expr` 的方法调用 (扩展表达式)

更符合 Polars 表达式链式调用风格的方式是，将 UDF 作为 `pl.Expr` 对象的方法来调用。这需要您的 Python 包中包含一个 `extension.py` 文件，该文件负责将 Rust UDF 注册为 `pl.Expr` 的方法。

在 `expression_lib/expression_lib/extension.py` 中，您会看到类似以下的代码（简化版）：

```python
# expression_lib/expression_lib/extension.py
import polars as pl
from polars.utils.udfs import _get_udf_namespace_functions

# 导入 Rust 模块
from expression_lib import language, dist, date_util, panic

# 注册 UDF 到 Polars 表达式命名空间
_get_udf_namespace_functions(
    pl.Expr,
    "language",
    [
        language.pig_latinnify,
        language.append_args,
    ],
)

_get_udf_namespace_functions(
    pl.Expr,
    "dist",
    [
        dist.hamming_distance,
        dist.jaccard_similarity,
        dist.haversine,
    ],
)

_get_udf_namespace_functions(
    pl.Expr,
    "date_util",
    [
        date_util.is_leap_year,
        date_util.change_time_zone,
    ],
)

_get_udf_namespace_functions(
    pl.Expr,
    "panic",
    [
        panic.panic,
    ],
)
```
通过导入 `expression_lib.extension` 模块，这些 UDF 就会被注册到 `pl.Expr` 的相应命名空间下。

**示例 (`run.py` 中的第二种调用方式)**:

```python
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
)
print(out)
```
在这种方式中，您首先使用 `pl.col("column_name")` 创建一个表达式，然后通过 `.` 运算符访问注册的 UDF 命名空间（例如 `.language`、`.dist`），再调用具体的 UDF 方法。这种方式更符合 Polars 的表达式范式，并且通常更推荐。

### 4.4 传递参数

无论是直接函数调用还是表达式方法调用，传递参数的方式是相似的：

*   **位置参数**: 对应于 Rust UDF 函数签名中的 `inputs: &[Series]`。在 Python 中，您传递列名字符串或 `pl.Expr` 对象作为位置参数。
*   **关键字参数**: 对应于 Rust UDF 函数签名中的 `kwargs: MyKwargs`。在 Python 中，您传递与 Rust 结构体字段名匹配的关键字参数。这些参数会被 `pyo3-polars` 序列化为 JSON 或 MessagePack，并在 Rust 端反序列化为 `MyKwargs` 结构体。

**示例**:

```python
# 传递位置参数 "names" 和关键字参数 capitalize=True
pig_latin_cap=language.pig_latinnify("names", capitalize=True),

# 传递位置参数 "names" 和多个关键字参数
appended_args=pl.col("names").language.append_args(
    float_arg=11.234,
    integer_arg=93,
    boolean_arg=False,
    string_arg="example",
),
```

## 5. 示例分析

`run.py` 脚本提供了多个示例，演示了如何使用 Rust UDF 处理不同类型的数据和传递不同类型的参数。

首先，脚本创建了一个包含多种数据类型的 Polars DataFrame：

```python
import polars as pl
from datetime import date, datetime, timezone
from expression_lib import language, dist, date_util, panic

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
    }
)
```

### 5.1 直接函数调用示例

第一个 `with_columns` 块展示了直接将 UDF 作为函数调用的方式：

```python
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
)
print(out)
```

*   `pig_latin=language.pig_latinnify("names")`: 调用 `pig_latinnify` UDF，将 "names" 列转换为 Pig Latin 格式。
*   `pig_latin_cap=language.pig_latinnify("names", capitalize=True)`: 再次调用 `pig_latinnify`，并传递 `capitalize=True` 关键字参数，使结果首字母大写。
*   `hamming_dist=dist.hamming_distance("names", "pig_latin")`: 计算 "names" 和 "pig_latin" 列之间的海明距离。
*   `jaccard_sim=dist.jaccard_similarity("dist_a", "dist_b")`: 计算 "dist_a" 和 "dist_b" 列表列之间的 Jaccard 相似度。
*   `haversine=dist.haversine("start_lat", "start_lon", "end_lat", "end_lon")`: 计算 Haversine 距离，需要四个浮点数输入列。
*   `leap_year=date_util.is_leap_year("dates")`: 判断 "dates" 列中的日期是否为闰年。
*   `new_tz=date_util.change_time_zone("datetime")`: 更改 "datetime" 列的时区。
*   `appended_args=language.append_args(...)`: 演示了如何向 Rust UDF 传递多种类型的关键字参数（浮点数、整数、布尔值、字符串），这些参数在 Rust 端被反序列化并用于构建新的字符串。

### 5.2 表达式方法调用示例

第二个 `with_columns` 块展示了将 UDF 作为 `pl.Expr` 方法调用的方式。这需要先导入 `expression_lib.extension` 模块来注册这些方法。

```python
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
)
print(out)
```
这里的调用方式与直接函数调用类似，但通过 `pl.col("column_name").namespace.udf_method()` 的链式调用，使得代码更具可读性和 Polars 风格。

### 5.3 错误处理示例

`run.py` 还包含了错误处理的示例，演示了当传递错误类型的参数或 Rust UDF 发生 panic 时，Python 端如何捕获 `polars.exceptions.ComputeError`。

```python
# Tests we can return errors from FFI by passing wrong types.
try:
    out.with_columns(
        appended_args=pl.col("names").language.append_args(
            float_arg=True, # 错误类型：期望浮点数，实际为布尔值
            integer_arg=True, # 错误类型：期望整数，实际为布尔值
            boolean_arg=True,
            string_arg="example",
        )
    )
except pl.exceptions.ComputeError as e:
    assert "the plugin failed with message" in str(e)


try:
    out.with_columns(pl.col("names").panic.panic()) # 调用会 panic 的 UDF
except pl.exceptions.ComputeError as e:
    assert "the plugin panicked" in str(e)

print("finished")
```

## 6. 总结

通过本指南，您应该已经了解了如何使用 `pyo3-polars` 库在 Rust 中实现高性能的 Polars UDF，并将其无缝集成到 Python 环境中。

**关键要点包括**:

*   **`pyo3-polars` 库**: 提供了 `#[polars_expr]` 宏和必要的类型，简化了 Rust 和 Polars 之间的交互。
*   **`#[polars_expr]` 宏**: 核心工具，用于将 Rust 函数转换为 Polars 表达式。
*   **函数签名**: 遵循 `fn my_udf(inputs: &[Series], kwargs: MyKwargs) -> PolarsResult<Series>` 模式。
*   **输出类型推断**: 使用 `output_type` 或 `output_type_func` 来指定或动态推断输出类型。
*   **数据处理**: 从 `Series` 中提取 `ChunkedArray`，进行数据处理，然后重新封装为 `Series`。
*   **自定义参数**: 通过 `#[derive(Deserialize)]` 结构体接收 Python 端的关键字参数。
*   **并行计算**: 利用 `CallerContext` 和 Polars 线程池实现并行处理。
*   **错误和 Panic 处理**: Rust 端的错误和 panic 会被转换为 Python 端的 `polars.exceptions.ComputeError`。
*   **Python 绑定**: 使用 `maturin develop` 进行编译和安装，并通过直接函数调用或 `pl.Expr` 方法调用来使用 UDF。

通过遵循这些模式和最佳实践，您可以为 Polars 扩展强大的自定义功能，从而在数据处理任务中获得更高的灵活性和性能。
这些 `try-except` 块验证了 `pyo3-polars` 能够正确地将 Rust 端的错误和 panic 传播到 Python 端，并以 `ComputeError` 的形式抛出。
```
</content>
