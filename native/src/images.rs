//! Dedicated codec libraries; FFmpeg is reserved for audio/video.
use numpy::{ndarray::Array, IntoPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::ffi::{c_char, c_int, c_void, CStr};

#[repr(C)]
struct Output {
    data: *mut c_void,
    frames: usize,
    height: usize,
    width: usize,
    channels: usize,
    bytes: usize,
    bits: c_int,
    orientation: c_int,
    animated: c_int,
    error: [c_char; 256],
}
impl Drop for Output {
    fn drop(&mut self) {
        unsafe { libc::free(self.data) };
    }
}
extern "C" {
    fn tc_decode_image(
        data: *const u8,
        size: usize,
        codec: c_int,
        mode: c_int,
        depth: c_int,
        threads: c_int,
        out: *mut Output,
    ) -> c_int;
}

#[pyfunction]
pub fn decode_image(
    py: Python<'_>,
    data: &[u8],
    codec: i32,
    mode: i32,
    depth: i32,
    threads: i32,
) -> PyResult<(PyObject, i32, bool)> {
    if !(0..=4).contains(&codec)
        || !(0..=4).contains(&mode)
        || ![0, 8, 16].contains(&depth)
        || threads < 1
    {
        return Err(PyValueError::new_err("invalid native image options"));
    }
    if codec == 1 {
        return Ok((crate::png_image::decode(py, data, mode, depth)?, 1, false));
    }
    // Only the input slice crosses the GIL boundary. Output pointers stay on this thread.
    let (pixels, shape, bits, orientation, animated) = py.allow_threads(|| {
        let mut out: Output = unsafe { std::mem::zeroed() };
        let ok = unsafe {
            tc_decode_image(
                data.as_ptr(),
                data.len(),
                codec,
                mode,
                depth,
                threads,
                &mut out,
            )
        };
        if ok == 0 {
            return Err(PyRuntimeError::new_err(
                unsafe { CStr::from_ptr(out.error.as_ptr()) }
                    .to_string_lossy()
                    .into_owned(),
            ));
        }
        let shape = (out.frames, out.height, out.width, out.channels);
        let bytes =
            unsafe { std::slice::from_raw_parts(out.data.cast::<u8>(), out.bytes) }.to_vec();
        Ok((bytes, shape, out.bits, out.orientation, out.animated != 0))
    })?;
    let array = if bits == 16 {
        let values: Vec<u16> = pixels
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect();
        Array::from_shape_vec(shape, values)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .into_pyarray(py)
            .into_any()
            .unbind()
    } else {
        Array::from_shape_vec(shape, pixels)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .into_pyarray(py)
            .into_any()
            .unbind()
    };
    Ok((array, orientation, animated))
}
