//! Still/animated images use sequential decoding, without video timestamps or seeking.
use numpy::{ndarray::Array, IntoPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::ffmpeg::{self, Pixels, Source};

#[pyfunction]
#[pyo3(signature = (data, channels=0, threads=1))]
pub fn decode_image(
    py: Python<'_>,
    data: &[u8],
    channels: usize,
    threads: i32,
) -> PyResult<PyObject> {
    if channels > 4 || threads < 1 {
        return Err(PyValueError::new_err(
            "invalid image channels or thread count",
        ));
    }
    let source = Source::Bytes(data.to_vec());
    let images = py
        .allow_threads(|| ffmpeg::Decoder::open(source, false, None, threads)?.images(channels))
        .map_err(ffmpeg::Error::into_py)?;
    let shape = (images.frames, images.height, images.width, images.channels);
    macro_rules! array {
        ($data:expr) => {
            Array::from_shape_vec(shape, $data)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .into_pyarray(py)
                .into_any()
                .unbind()
        };
    }
    Ok(match images.pixels {
        Pixels::U8(data) => array!(data),
        Pixels::U16(data) => array!(data),
        Pixels::F32(_) => unreachable!(),
    })
}
