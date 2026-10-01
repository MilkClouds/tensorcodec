//! Only the FFmpeg boundary is native. Public API and playback planning live in Python.

mod ffmpeg;

use numpy::{ndarray::Array, IntoPyArray};
use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

#[pyclass]
struct Decoder {
    inner: Option<ffmpeg::Decoder>,
}

#[pymethods]
impl Decoder {
    #[new]
    #[pyo3(signature = (source, media_type, stream_index=None, threads=1))]
    fn new(
        py: Python<'_>,
        source: &Bound<'_, PyAny>,
        media_type: &str,
        stream_index: Option<i32>,
        threads: i32,
    ) -> PyResult<Self> {
        let input = if let Ok(path) = source.extract::<String>() {
            ffmpeg::Source::Path(path)
        } else if let Ok(data) = source.downcast::<PyBytes>() {
            ffmpeg::Source::Bytes(data.as_bytes().to_vec())
        } else if source.hasattr("read")? && source.hasattr("seek")? {
            ffmpeg::Source::File(source.clone().unbind())
        } else {
            return Err(PyValueError::new_err(
                "source must be a path/URL or encoded bytes",
            ));
        };
        let is_audio = match media_type {
            "video" => false,
            "audio" => true,
            _ => return Err(PyValueError::new_err("media_type must be video or audio")),
        };
        let inner = py
            .allow_threads(move || ffmpeg::Decoder::open(input, is_audio, stream_index, threads))
            .map_err(ffmpeg::Error::into_py)?;
        Ok(Self { inner: Some(inner) })
    }

    #[pyo3(signature = (apply_rotation=true))]
    fn metadata<'py>(&self, py: Python<'py>, apply_rotation: bool) -> PyResult<Bound<'py, PyDict>> {
        self.inner
            .as_ref()
            .ok_or_else(closed)?
            .metadata(py, apply_rotation)
    }

    fn scan(&mut self, py: Python<'_>) -> PyResult<Vec<(i64, i64, bool)>> {
        let inner = self.inner.as_mut().ok_or_else(closed)?;
        py.allow_threads(|| inner.scan())
            .map_err(ffmpeg::Error::into_py)
    }

    fn decode_video(
        &mut self,
        py: Python<'_>,
        targets: Vec<(i64, i64)>,
        output_dtype: &str,
        exact: bool,
    ) -> PyResult<(PyObject, Vec<f64>, Vec<f64>)> {
        self.video_request(
            py,
            ffmpeg::VideoRequest::Frames { targets, exact },
            output_dtype,
        )
    }

    fn decode_timestamps(
        &mut self,
        py: Python<'_>,
        seconds: Vec<f64>,
        output_dtype: &str,
    ) -> PyResult<(PyObject, Vec<f64>, Vec<f64>)> {
        self.video_request(py, ffmpeg::VideoRequest::Timestamps(seconds), output_dtype)
    }

    #[pyo3(signature = (sample_rate, channels, stop=None))]
    fn decode_audio(
        &mut self,
        py: Python<'_>,
        sample_rate: i32,
        channels: i32,
        stop: Option<f64>,
    ) -> PyResult<(PyObject, f64)> {
        let inner = self.inner.as_mut().ok_or_else(closed)?;
        let output = py
            .allow_threads(|| inner.audio(sample_rate, channels, stop))
            .map_err(ffmpeg::Error::into_py)?;
        let array = Array::from_shape_vec((channels as usize, output.samples), output.data)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .into_pyarray(py)
            .into_any()
            .unbind();
        Ok((array, output.pts))
    }

    fn close(&mut self) {
        self.inner.take();
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(source) = self.inner.as_ref().and_then(|inner| inner.file_source()) {
            visit.call(source)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.inner.take();
    }
}

fn closed() -> PyErr {
    PyRuntimeError::new_err("decoder is closed")
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Decoder>()?;
    module.add("ffmpeg_version", ffmpeg::version())?;
    Ok(())
}

impl Decoder {
    fn video_request(
        &mut self,
        py: Python<'_>,
        request: ffmpeg::VideoRequest,
        output_dtype: &str,
    ) -> PyResult<(PyObject, Vec<f64>, Vec<f64>)> {
        let inner = self.inner.as_mut().ok_or_else(closed)?;
        let dtype = match output_dtype {
            "native" => ffmpeg::OutputDtype::Native,
            "uint8" => ffmpeg::OutputDtype::U8,
            "uint16" => ffmpeg::OutputDtype::U16,
            "float32" => ffmpeg::OutputDtype::F32,
            _ => return Err(PyValueError::new_err("invalid video output dtype")),
        };
        let output = py
            .allow_threads(|| inner.video(request, dtype))
            .map_err(ffmpeg::Error::into_py)?;
        let shape = (
            output.pts.len(),
            output.height,
            output.width,
            output.channels,
        );
        let array = match output.pixels {
            ffmpeg::Pixels::U8(data) => Array::from_shape_vec(shape, data)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .into_pyarray(py)
                .into_any()
                .unbind(),
            ffmpeg::Pixels::F32(data) => Array::from_shape_vec(shape, data)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .into_pyarray(py)
                .into_any()
                .unbind(),
            ffmpeg::Pixels::U16(data) => Array::from_shape_vec(shape, data)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .into_pyarray(py)
                .into_any()
                .unbind(),
        };
        Ok((array, output.pts, output.durations))
    }
}
