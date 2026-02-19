# Architecture: shrew-python

`shrew-python` provides Python bindings for Shrew, enabling users to leverage the framework's performance with Python's ease of use. It uses `PyO3` to wrap Rust types and expose them as a native Python extension module.

## Core Concepts

- **PyO3 Integration**: Maps Rust structs (`Tensor`, `Optimizer`, `Module`) to Python classes.
- **NumPy Interop**: Zero-copy (where possible) conversion between Shrew Tensors and NumPy arrays.
- **Global Interpreter Lock**: carefully manages GIL release during heavy computations to allow multi-threading.
- **Exceptions**: Translates Shrew's `Result` types into standard Python exceptions (`ValueError`, `RuntimeError`).

## File Structure

| File | Description | Lines of Code |
| :--- | :--- | :--- |
| `lib.rs` | The massive single-file implementation of the bindings. It defines the Python module `shrew_python`, the `Tensor` class, operator overloads `__add__`, `__mul__`, and the `Executor` class for running `.sw` models. | 2332 |
