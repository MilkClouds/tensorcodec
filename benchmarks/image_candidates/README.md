# Experimental image decoders

A separate PyO3 module for comparing `image` and `libjpeg-turbo-rs` with the
production decoder. It only exercises still RGB uint8 decoding without EXIF
orientation; it is not an implementation of TensorCodec's full image contract.
Its crates do not enter the production dependency graph.

Build with `maturin develop --release --locked --manifest-path
benchmarks/image_candidates/Cargo.toml --uv` inside the project's uv environment.
Run `benchmarks/image_decode.py --candidates --output results.json`.
See `docs/image-backends.md` for measurement scope and the recorded JSON reports.
