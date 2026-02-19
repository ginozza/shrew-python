# shrew-python

Python bindings for [Shrew](https://github.com/ginozza/shrew), a modular deep learning framework written in Rust.

<p align="center">
  <a href="https://pypi.org/project/shrew-python/"><img src="https://img.shields.io/pypi/v/shrew-python.svg?color=3775A9&logo=python&logoColor=white" alt="PyPI"></a>
</p>

## Features

- **High Performance**: Native Rust implementation with minimal Python overhead.
- **GPU Support**: CUDA acceleration via `shrew-cuda` (if valid CUDA toolkit is present).
- **Declarative Models**: Full support for Shrew's `.sw` intermediate representation.
- **Autograd**: Reverse-mode automatic differentiation.
- **Interoperability**: Zero-copy tensor conversion from/to NumPy.

## Installation

```bash
pip install shrew-python
```

## Usage

```python
import shrew_python as shrew

# Create tensors
x = shrew.tensor([1.0, 2.0, 3.0])
y = shrew.tensor([4.0, 5.0, 6.0])

# Operations
z = x + y
print(z)  # Tensor([5.0, 7.0, 9.0], dtype=F64, dev=Cpu)

# Load a .sw model
executor = shrew.Executor.load("my_model.sw")
result = executor.run("forward", {"input": x})
```

## Building from Source

Requires [Rust](https://rustup.rs/) and [Maturin](https://github.com/PyO3/maturin).

```bash
git clone https://github.com/ginozza/shrew
cd shrew
maturin develop --manifest-path crates/shrew-python/Cargo.toml --release
```

## License

Apache-2.0
