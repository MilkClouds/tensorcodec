use numpy::{ndarray::Array, IntoPyArray};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
#[pyfunction]
fn decode(py: Python<'_>, data: &[u8]) -> PyResult<PyObject> {
    let (w, h, bytes) = py.allow_threads(|| {
        let decoded = image::load_from_memory(data)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .into_rgb8();
        Ok::<_, PyErr>((
            decoded.width() as usize,
            decoded.height() as usize,
            decoded.into_raw(),
        ))
    })?;
    Ok(Array::from_shape_vec((h, w, 3), bytes)
        .unwrap()
        .into_pyarray(py)
        .into_any()
        .unbind())
}
#[pyfunction]
fn decode_turbo(py: Python<'_>, data: &[u8]) -> PyResult<PyObject> {
    let decoded = py
        .allow_threads(|| libjpeg_turbo_rs::decompress(data))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(
        Array::from_shape_vec((decoded.height, decoded.width, 3), decoded.data)
            .unwrap()
            .into_pyarray(py)
            .into_any()
            .unbind(),
    )
}
#[pymodule]
fn _image_candidate(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(decode, module)?)?;
    module.add_function(wrap_pyfunction!(decode_turbo, module)?)?;
    Ok(())
}
