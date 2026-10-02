//! PNG sample decoding in safe Rust, with TorchCodec/libpng mode semantics.
use numpy::{ndarray::Array, IntoPyArray};
use png::{BitDepth, ColorType, Transformations};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use std::io::Cursor;

struct Decoded {
    pixels: Vec<u8>,
    width: usize,
    height: usize,
    channels: usize,
    bits: u8,
}
fn read(data: &[u8], mode: i32, depth: i32) -> Result<Decoded, String> {
    let mut decoder = png::Decoder::new(Cursor::new(data));
    decoder.set_transformations(Transformations::EXPAND);
    let mut reader = decoder.read_info().map_err(|e| e.to_string())?;
    let source = reader.info();
    if source.animation_control.is_some() {
        return Err("animated PNG decoding is unsupported".into());
    }
    let native_channels = source.color_type.samples();
    let strip_transparency =
        mode == 0 && source.color_type != ColorType::Indexed && source.trns.is_some();
    let size = reader
        .output_buffer_size()
        .ok_or("PNG dimensions overflow")?;
    let mut pixels = Vec::new();
    pixels.try_reserve_exact(size).map_err(|e| e.to_string())?;
    pixels.resize(size, 0);
    let frame = reader.next_frame(&mut pixels).map_err(|e| e.to_string())?;
    reader.finish().map_err(|e| e.to_string())?;
    pixels.truncate(frame.buffer_size());
    let src_channels = frame.color_type.samples();
    let src_bits = if frame.bit_depth == BitDepth::Sixteen {
        16
    } else {
        8
    };
    let bits = if depth == 0 { src_bits } else { depth as u8 };
    let channels = if mode != 0 {
        mode as usize
    } else if strip_transparency {
        native_channels
    } else {
        src_channels
    };
    let (width, height) = (frame.width as usize, frame.height as usize);
    if channels == src_channels && bits == 8 && src_bits == 8 {
        return Ok(Decoded {
            pixels,
            width,
            height,
            channels,
            bits,
        });
    }
    let length = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(channels))
        .and_then(|n| n.checked_mul((bits / 8) as usize))
        .ok_or("PNG output dimensions overflow")?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(length)
        .map_err(|e| e.to_string())?;
    let maximum = if src_bits == 16 { 65535 } else { 255 };
    for pixel in pixels.chunks_exact(src_channels * (src_bits / 8) as usize) {
        let mut values = [0u32; 4];
        for (i, value) in values.iter_mut().enumerate().take(src_channels) {
            *value = if src_bits == 16 {
                u16::from_be_bytes([pixel[2 * i], pixel[2 * i + 1]]) as u32
            } else {
                pixel[i] as u32
            };
        }
        let gray_source = src_channels <= 2;
        let alpha = if src_channels == 2 || src_channels == 4 {
            values[src_channels - 1]
        } else {
            maximum
        };
        let gray = if gray_source {
            values[0]
        } else {
            (values[0] * 9794
                + values[1] * 19234
                + values[2] * 3740
                + if src_bits == 16 { 16384 } else { 0 })
                >> 15
        };
        let converted = if channels <= 2 {
            [gray, alpha, 0, 0]
        } else if gray_source {
            [values[0], values[0], values[0], alpha]
        } else {
            [values[0], values[1], values[2], alpha]
        };
        for value in &converted[..channels] {
            let value = match (src_bits, bits) {
                (8, 16) => value * 257,
                (16, 8) => (value + 128) / 257,
                _ => *value,
            };
            if bits == 16 {
                output.extend_from_slice(&(value as u16).to_ne_bytes());
            } else {
                output.push(value as u8);
            }
        }
    }
    Ok(Decoded {
        pixels: output,
        width,
        height,
        channels,
        bits,
    })
}

pub fn decode(py: Python<'_>, data: &[u8], mode: i32, depth: i32) -> PyResult<PyObject> {
    let output = py
        .allow_threads(|| read(data, mode, depth))
        .map_err(PyRuntimeError::new_err)?;
    let shape = (1, output.height, output.width, output.channels);
    if output.bits == 16 {
        let values: Vec<u16> = output
            .pixels
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect();
        Ok(Array::from_shape_vec(shape, values)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .into_pyarray(py)
            .into_any()
            .unbind())
    } else {
        Ok(Array::from_shape_vec(shape, output.pixels)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .into_pyarray(py)
            .into_any()
            .unbind())
    }
}
