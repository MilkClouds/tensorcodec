//! Still/animated images use sequential decoding, without video timestamps or seeking.
use numpy::{
    ndarray::{Array, Axis},
    IntoPyArray,
};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::ffmpeg::{self, Pixels, Source};

#[pyfunction]
#[pyo3(signature = (data, channels=0, threads=1, apply_rotation=false))]
pub fn decode_image(
    py: Python<'_>,
    data: &[u8],
    channels: usize,
    threads: i32,
    apply_rotation: bool,
) -> PyResult<PyObject> {
    if channels > 4 || threads < 1 {
        return Err(PyValueError::new_err(
            "invalid image channels or thread count",
        ));
    }
    let source = Source::Bytes(data.to_vec());
    let mut decoder = py
        .allow_threads(|| ffmpeg::Decoder::open(source, false, None, threads))
        .map_err(ffmpeg::Error::into_py)?;
    let turns = if apply_rotation {
        let metadata = decoder.metadata(py, true)?;
        let rotation = metadata
            .get_item("rotation")?
            .unwrap()
            .extract::<Option<f64>>()?
            .unwrap_or(0.);
        let turns = (rotation / 90.).round();
        if (rotation - turns * 90.).abs() > 1e-4 {
            return Err(PyNotImplementedError::new_err(
                "non-right-angle image rotation is unsupported",
            ));
        }
        (turns as i32).rem_euclid(4)
    } else {
        0
    };
    let images = py
        .allow_threads(|| decoder.images(channels))
        .map_err(ffmpeg::Error::into_py)?;
    let shape = (images.frames, images.height, images.width, images.channels);
    macro_rules! array {
        ($data:expr) => {{
            let mut array = Array::from_shape_vec(shape, $data)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            if turns % 2 == 1 {
                array.swap_axes(1, 2);
            }
            if turns == 1 || turns == 2 {
                array.invert_axis(Axis(1));
            }
            if turns == 2 || turns == 3 {
                array.invert_axis(Axis(2));
            }
            array.into_pyarray(py).into_any().unbind()
        }};
    }
    Ok(match images.pixels {
        Pixels::U8(data) => array!(data),
        Pixels::U16(data) => array!(data),
        Pixels::F32(_) => unreachable!(),
    })
}
