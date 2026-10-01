//! FFmpeg handles are owned here and never exposed to Python. Each decoder is
//! exclusively borrowed by PyO3 during a native operation; no global mutable state.
use ffmpeg_sys_next as av;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::BTreeMap;
use std::ffi::{c_void, CStr, CString};
use std::ptr;

pub struct Error(pub String, pub bool);
impl Error {
    pub fn into_py(self) -> PyErr {
        if self.1 {
            PyValueError::new_err(self.0)
        } else {
            PyRuntimeError::new_err(self.0)
        }
    }
}
type Result<T> = std::result::Result<T, Error>;
fn failure(message: impl Into<String>) -> Error {
    Error(message.into(), false)
}
fn check(code: i32, operation: &str) -> Result<()> {
    if code >= 0 {
        return Ok(());
    }
    let mut text: [std::ffi::c_char; 256] = [0; 256];
    unsafe {
        av::av_strerror(code, text.as_mut_ptr(), text.len());
    }
    Err(failure(format!(
        "{operation}: {}",
        unsafe { CStr::from_ptr(text.as_ptr()) }.to_string_lossy()
    )))
}
fn name(pointer: *const std::ffi::c_char) -> Option<String> {
    if pointer.is_null() {
        None
    } else {
        Some(
            unsafe { CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned(),
        )
    }
}
pub fn version() -> String {
    name(unsafe { av::av_version_info() }).unwrap_or_default()
}
pub enum Source {
    Path(String),
    Bytes(Vec<u8>),
    File(Py<PyAny>),
}
struct Memory {
    data: Vec<u8>,
    position: usize,
}
enum Reader {
    Bytes(Memory),
    File(Py<PyAny>),
}
unsafe extern "C" fn read_memory(opaque: *mut c_void, buffer: *mut u8, size: i32) -> i32 {
    if size <= 0 {
        return -22;
    }
    let reader = &mut *(opaque as *mut Reader);
    let input = match reader {
        Reader::Bytes(input) => input,
        Reader::File(source) => {
            return Python::with_gil(|py| -> PyResult<i32> {
                let data = source.bind(py).call_method1("read", (size,))?;
                let data = data.downcast::<pyo3::types::PyBytes>()?.as_bytes();
                if data.len() > size as usize {
                    return Ok(-5);
                }
                if data.is_empty() {
                    return Ok(av::AVERROR_EOF);
                }
                ptr::copy_nonoverlapping(data.as_ptr(), buffer, data.len());
                Ok(data.len() as i32)
            })
            .unwrap_or(-5)
        }
    };
    let count = (size as usize).min(input.data.len().saturating_sub(input.position));
    if count == 0 {
        return av::AVERROR_EOF;
    }
    ptr::copy_nonoverlapping(input.data.as_ptr().add(input.position), buffer, count);
    input.position += count;
    count as i32
}
unsafe extern "C" fn seek_memory(opaque: *mut c_void, offset: i64, whence: i32) -> i64 {
    let reader = &mut *(opaque as *mut Reader);
    let input = match reader {
        Reader::Bytes(input) => input,
        Reader::File(source) => {
            return Python::with_gil(|py| -> PyResult<i64> {
                let source = source.bind(py);
                if whence & av::AVSEEK_SIZE != 0 {
                    let position: i64 = source.call_method1("seek", (0, 1))?.extract()?;
                    let size: i64 = source.call_method1("seek", (0, 2))?.extract()?;
                    source.call_method1("seek", (position, 0))?;
                    Ok(size)
                } else {
                    source
                        .call_method1("seek", (offset, whence & !av::AVSEEK_FORCE))?
                        .extract()
                }
            })
            .unwrap_or(-5)
        }
    };
    if whence & av::AVSEEK_SIZE != 0 {
        return input.data.len() as i64;
    }
    let base = match whence & !av::AVSEEK_FORCE {
        0 => 0,
        1 => input.position as i64,
        2 => input.data.len() as i64,
        _ => return -22,
    };
    let Some(position) = base.checked_add(offset) else {
        return -22;
    };
    if position < 0 || position > input.data.len() as i64 {
        return -22;
    }
    input.position = position as usize;
    position
}

pub struct Decoder {
    format: *mut av::AVFormatContext,
    codec: *mut av::AVCodecContext,
    packet: *mut av::AVPacket,
    frame: *mut av::AVFrame,
    rgb_frame: *mut av::AVFrame,
    scale: *mut av::SwsContext,
    scale_config: Option<(i32, i32, i32, i32)>,
    io: *mut av::AVIOContext,
    memory: Option<Box<Reader>>,
    index: i32,
    time_base: av::AVRational,
    audio: bool,
    draining: bool,
}
// SAFETY: all pointers are uniquely owned. PyO3's mutable borrow plus the Python
// wrapper's RLock serialize access. Custom AVIO callbacks acquire the GIL before
// accessing Python file objects; native operations release it before calling FFmpeg.
unsafe impl Send for Decoder {}
unsafe impl Sync for Decoder {}

impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe {
            av::sws_freeContext(self.scale);
            av::av_frame_free(&mut self.frame);
            av::av_frame_free(&mut self.rgb_frame);
            av::av_packet_free(&mut self.packet);
            av::avcodec_free_context(&mut self.codec);
            av::avformat_close_input(&mut self.format);
            if !self.io.is_null() {
                av::av_free((*self.io).buffer as *mut c_void);
                (*self.io).buffer = ptr::null_mut();
                av::avio_context_free(&mut self.io);
            }
        }
    }
}

impl Decoder {
    pub fn file_source(&self) -> Option<&Py<PyAny>> {
        match self.memory.as_deref() {
            Some(Reader::File(source)) => Some(source),
            _ => None,
        }
    }

    pub fn open(source: Source, audio: bool, index: Option<i32>, threads: i32) -> Result<Self> {
        let mut this = Self {
            format: ptr::null_mut(),
            codec: ptr::null_mut(),
            packet: ptr::null_mut(),
            frame: ptr::null_mut(),
            rgb_frame: ptr::null_mut(),
            scale: ptr::null_mut(),
            scale_config: None,
            io: ptr::null_mut(),
            memory: None,
            index: 0,
            time_base: av::AVRational { num: 0, den: 1 },
            audio,
            draining: false,
        };
        unsafe {
            let path = match source {
                Source::Path(path) => {
                    Some(CString::new(path).map_err(|_| Error("path contains NUL".into(), true))?)
                }
                input => {
                    let reader = match input {
                        Source::Bytes(data) => Reader::Bytes(Memory { data, position: 0 }),
                        Source::File(source) => {
                            Python::with_gil(|py| {
                                source.bind(py).call_method1("seek", (0, 0)).map(|_| ())
                            })
                            .map_err(|error| failure(error.to_string()))?;
                            Reader::File(source)
                        }
                        Source::Path(_) => unreachable!(),
                    };
                    this.memory = Some(Box::new(reader));
                    let buffer = av::av_malloc(32768) as *mut u8;
                    if buffer.is_null() {
                        return Err(failure("cannot allocate input buffer"));
                    }
                    let opaque = &mut **this.memory.as_mut().unwrap() as *mut Reader as *mut c_void;
                    this.io = av::avio_alloc_context(
                        buffer,
                        32768,
                        0,
                        opaque,
                        Some(read_memory),
                        None,
                        Some(seek_memory),
                    );
                    if this.io.is_null() {
                        av::av_free(buffer as *mut c_void);
                        return Err(failure("cannot allocate input context"));
                    }
                    this.format = av::avformat_alloc_context();
                    if this.format.is_null() {
                        return Err(failure("cannot allocate format context"));
                    }
                    (*this.format).pb = this.io;
                    (*this.format).flags |= av::AVFMT_FLAG_CUSTOM_IO;
                    None
                }
            };
            check(
                av::avformat_open_input(
                    &mut this.format,
                    path.as_ref().map_or(ptr::null(), |p| p.as_ptr()),
                    ptr::null_mut(),
                    ptr::null_mut(),
                ),
                "open input",
            )?;
            check(
                av::avformat_find_stream_info(this.format, ptr::null_mut()),
                "read stream metadata",
            )?;
            let media_type = if audio {
                av::AVMediaType::AVMEDIA_TYPE_AUDIO
            } else {
                av::AVMediaType::AVMEDIA_TYPE_VIDEO
            };
            this.index = match index {
                Some(i) => i,
                None => {
                    av::av_find_best_stream(this.format, media_type, -1, -1, ptr::null_mut(), 0)
                }
            };
            if this.index < 0 || this.index >= (*this.format).nb_streams as i32 {
                return Err(Error(
                    "no valid stream of the requested media type".into(),
                    true,
                ));
            }
            let stream = this.stream();
            let params = (*stream).codecpar;
            if (*params).codec_type != media_type {
                return Err(Error("stream has the wrong media type".into(), true));
            }
            this.time_base = (*stream).time_base;
            if this.time_base.den <= 0 || this.time_base.num <= 0 {
                return Err(failure("invalid stream time base"));
            }
            let codec = av::avcodec_find_decoder((*params).codec_id);
            if codec.is_null() {
                return Err(failure("FFmpeg has no decoder for this codec"));
            }
            this.codec = av::avcodec_alloc_context3(codec);
            if this.codec.is_null() {
                return Err(failure("cannot allocate codec context"));
            }
            check(
                av::avcodec_parameters_to_context(this.codec, params),
                "configure decoder",
            )?;
            (*this.codec).thread_count = threads;
            (*this.codec).pkt_timebase = this.time_base;
            check(
                av::avcodec_open2(this.codec, codec, ptr::null_mut()),
                "open decoder",
            )?;
            this.packet = av::av_packet_alloc();
            this.frame = av::av_frame_alloc();
            this.rgb_frame = av::av_frame_alloc();
            if this.packet.is_null() || this.frame.is_null() || this.rgb_frame.is_null() {
                return Err(failure("cannot allocate decoding buffers"));
            }
        }
        Ok(this)
    }

    unsafe fn stream(&self) -> *mut av::AVStream {
        *(*self.format).streams.add(self.index as usize)
    }
    fn seconds(&self, pts: i64) -> f64 {
        pts as f64 * self.time_base.num as f64 / self.time_base.den as f64
    }
    fn begin_pts(&self) -> i64 {
        let start = unsafe { (*self.stream()).start_time };
        if start == av::AV_NOPTS_VALUE {
            0
        } else {
            start
        }
    }
    fn seek(&mut self, pts: i64) -> Result<()> {
        unsafe {
            check(
                av::av_seek_frame(self.format, self.index, pts, av::AVSEEK_FLAG_BACKWARD),
                "seek",
            )?;
            av::avcodec_flush_buffers(self.codec);
            av::av_packet_unref(self.packet);
            av::av_frame_unref(self.frame);
        }
        self.draining = false;
        Ok(())
    }

    pub fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let data = PyDict::new(py);
        unsafe {
            let stream = &*self.stream();
            let params = &*stream.codecpar;
            data.set_item("stream_index", self.index)?;
            data.set_item("media_type", if self.audio { "audio" } else { "video" })?;
            data.set_item("time_base_num", self.time_base.num)?;
            data.set_item("time_base_den", self.time_base.den)?;
            data.set_item("codec", name(av::avcodec_get_name(params.codec_id)))?;
            data.set_item("bit_rate", params.bit_rate as f64)?;
            data.set_item(
                "duration_seconds_from_header",
                if stream.duration > 0 {
                    Some(self.seconds(stream.duration))
                } else {
                    None
                },
            )?;
            data.set_item(
                "container_duration",
                if (*self.format).duration > 0 {
                    Some((*self.format).duration as f64 / av::AV_TIME_BASE as f64)
                } else {
                    None
                },
            )?;
            data.set_item(
                "begin_stream_seconds_from_header",
                if stream.start_time == av::AV_NOPTS_VALUE {
                    None
                } else {
                    Some(self.seconds(stream.start_time))
                },
            )?;
            if self.audio {
                data.set_item("sample_rate", params.sample_rate)?;
                data.set_item("num_channels", params.ch_layout.nb_channels)?;
                let format: av::AVSampleFormat = std::mem::transmute(params.format);
                data.set_item("sample_format", name(av::av_get_sample_fmt_name(format)))?;
            } else {
                data.set_item("width", params.width)?;
                data.set_item("height", params.height)?;
                data.set_item(
                    "num_frames_from_header",
                    if stream.nb_frames > 0 {
                        Some(stream.nb_frames)
                    } else {
                        None
                    },
                )?;
                let fps = av::av_q2d(stream.r_frame_rate);
                data.set_item(
                    "average_fps_from_header",
                    if fps > 0. { Some(fps) } else { None },
                )?;
                data.set_item(
                    "pixel_aspect_ratio",
                    (
                        params.sample_aspect_ratio.num,
                        params.sample_aspect_ratio.den,
                    ),
                )?;
                let pixel_format: av::AVPixelFormat = std::mem::transmute(params.format);
                data.set_item("pixel_format", name(av::av_get_pix_fmt_name(pixel_format)))?;
                data.set_item(
                    "color_space",
                    if params.color_space == av::AVColorSpace::AVCOL_SPC_UNSPECIFIED {
                        None
                    } else {
                        name(av::av_color_space_name(params.color_space))
                    },
                )?;
                data.set_item(
                    "color_primaries",
                    if params.color_primaries == av::AVColorPrimaries::AVCOL_PRI_UNSPECIFIED {
                        None
                    } else {
                        name(av::av_color_primaries_name(params.color_primaries))
                    },
                )?;
                data.set_item(
                    "color_transfer_characteristic",
                    if params.color_trc == av::AVColorTransferCharacteristic::AVCOL_TRC_UNSPECIFIED
                    {
                        None
                    } else {
                        name(av::av_color_transfer_name(params.color_trc))
                    },
                )?;
                let mut size = 0;
                let matrix = av::av_stream_get_side_data(
                    self.stream(),
                    av::AVPacketSideDataType::AV_PKT_DATA_DISPLAYMATRIX,
                    &mut size,
                );
                let rotation = if !matrix.is_null() && size >= 36 {
                    Some(-av::av_display_rotation_get(matrix as *const i32))
                } else {
                    None
                };
                data.set_item("rotation", rotation.filter(|r| r.is_finite() && *r != 0.))?;
            }
        }
        Ok(data)
    }

    pub fn scan(&mut self) -> Result<Vec<(i64, i64, bool)>> {
        self.seek(self.begin_pts())?;
        let mut frames = Vec::new();
        unsafe {
            loop {
                let code = av::av_read_frame(self.format, self.packet);
                if code == av::AVERROR_EOF {
                    break;
                }
                check(code, "scan packets")?;
                let packet = &*self.packet;
                if packet.stream_index == self.index && packet.flags & av::AV_PKT_FLAG_DISCARD == 0
                {
                    let pts = if packet.pts != av::AV_NOPTS_VALUE {
                        packet.pts
                    } else {
                        packet.dts
                    };
                    if pts == av::AV_NOPTS_VALUE {
                        return Err(failure("video packet has neither PTS nor DTS"));
                    }
                    frames.push((
                        pts,
                        packet.duration,
                        packet.flags & av::AV_PKT_FLAG_KEY != 0,
                    ));
                }
                av::av_packet_unref(self.packet);
            }
        }
        frames.sort_by_key(|frame| frame.0);
        self.seek(self.begin_pts())?;
        Ok(frames)
    }

    fn next(&mut self) -> Result<bool> {
        unsafe {
            av::av_frame_unref(self.frame);
            loop {
                let code = av::avcodec_receive_frame(self.codec, self.frame);
                if code >= 0 {
                    return Ok(true);
                }
                if code == av::AVERROR_EOF {
                    return Ok(false);
                }
                if code != -11 {
                    check(code, "receive decoded frame")?;
                }
                if self.draining {
                    return Ok(false);
                }
                loop {
                    let code = av::av_read_frame(self.format, self.packet);
                    if code == av::AVERROR_EOF {
                        self.draining = true;
                        check(
                            av::avcodec_send_packet(self.codec, ptr::null()),
                            "drain decoder",
                        )?;
                        break;
                    }
                    check(code, "read packet")?;
                    if (*self.packet).stream_index == self.index
                        && (self.audio || (*self.packet).flags & av::AV_PKT_FLAG_DISCARD == 0)
                    {
                        let code = av::avcodec_send_packet(self.codec, self.packet);
                        av::av_packet_unref(self.packet);
                        check(code, "send packet")?;
                        break;
                    }
                    av::av_packet_unref(self.packet);
                }
            }
        }
    }

    fn frame_pts(&self) -> Result<i64> {
        let frame = unsafe { &*self.frame };
        let pts = if frame.pts != av::AV_NOPTS_VALUE {
            frame.pts
        } else {
            frame.best_effort_timestamp
        };
        if pts == av::AV_NOPTS_VALUE {
            Err(failure("decoded frame has no timestamp"))
        } else {
            Ok(pts)
        }
    }

    pub fn video(
        &mut self,
        targets: Vec<(i64, i64)>,
        float_output: bool,
        exact: bool,
    ) -> Result<Video> {
        if self.audio {
            return Err(failure("cannot decode video from audio stream"));
        }
        let width = unsafe { (*self.codec).width } as usize;
        let height = unsafe { (*self.codec).height } as usize;
        let count = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| failure("frame is too large"))?;
        let total = count
            .checked_mul(targets.len())
            .ok_or_else(|| failure("batch is too large"))?;
        let mut pixels = vec![
            0u8;
            total
                .checked_mul(if float_output { 2 } else { 1 })
                .and_then(|n| n.checked_add(64))
                .ok_or_else(|| failure("batch is too large"))?
        ];
        let mut pts = vec![0.; targets.len()];
        let mut durations = vec![0.; targets.len()];
        let mut requests: BTreeMap<i64, (i64, Vec<usize>)> = BTreeMap::new();
        for (i, (target, key)) in targets.into_iter().enumerate() {
            requests
                .entry(target)
                .or_insert((key, Vec::new()))
                .1
                .push(i);
        }
        let mut active_key = None;
        let stride = count * if float_output { 2 } else { 1 };
        for (target, (key, positions)) in requests {
            if active_key != Some(key) {
                self.seek(key)?;
                active_key = Some(key);
            }
            let mut retried_from_beginning = false;
            loop {
                if !self.next()? {
                    return Err(failure(format!("no decoded frame at PTS {target}")));
                }
                let decoded_pts = self.frame_pts()?;
                if decoded_pts < target {
                    continue;
                }
                if exact && decoded_pts != target {
                    if retried_from_beginning {
                        return Err(failure(format!("no decoded frame at exact PTS {target}")));
                    }
                    self.seek(self.begin_pts())?;
                    active_key = None;
                    retried_from_beginning = true;
                    continue;
                }
                // Approximate mode intentionally picks the next decoded frame at/after target.
                let first = positions[0];
                unsafe {
                    let frame = &*self.frame;
                    if frame.width as usize != width || frame.height as usize != height {
                        return Err(failure("dynamic frame dimensions are unsupported"));
                    }
                    let input_format: av::AVPixelFormat = std::mem::transmute(frame.format);
                    let output_format = if float_output {
                        av::AVPixelFormat::AV_PIX_FMT_RGB48LE
                    } else {
                        av::AVPixelFormat::AV_PIX_FMT_RGB24
                    };
                    let config = (
                        frame.width,
                        frame.height,
                        input_format as i32,
                        output_format as i32,
                    );
                    if self.scale_config != Some(config) {
                        av::sws_freeContext(self.scale);
                        self.scale = av::sws_getContext(
                            frame.width,
                            frame.height,
                            input_format,
                            frame.width,
                            frame.height,
                            output_format,
                            0,
                            ptr::null_mut(),
                            ptr::null_mut(),
                            ptr::null(),
                        );
                        self.scale_config = Some(config);
                    }
                    if self.scale.is_null() {
                        return Err(failure("cannot initialize color conversion"));
                    }
                    let mut inverse = ptr::null_mut();
                    let mut table = ptr::null_mut();
                    let (
                        mut source_range,
                        mut destination_range,
                        mut brightness,
                        mut contrast,
                        mut saturation,
                    ) = (0, 0, 0, 0, 0);
                    check(
                        av::sws_getColorspaceDetails(
                            self.scale,
                            &mut inverse,
                            &mut source_range,
                            &mut table,
                            &mut destination_range,
                            &mut brightness,
                            &mut contrast,
                            &mut saturation,
                        ),
                        "read color conversion settings",
                    )?;
                    if frame.color_range != av::AVColorRange::AVCOL_RANGE_UNSPECIFIED {
                        source_range =
                            i32::from(frame.color_range == av::AVColorRange::AVCOL_RANGE_JPEG);
                    }
                    let coefficients = av::sws_getCoefficients(frame.colorspace as i32);
                    check(
                        av::sws_setColorspaceDetails(
                            self.scale,
                            coefficients,
                            source_range,
                            coefficients,
                            destination_range,
                            brightness,
                            contrast,
                            saturation,
                        ),
                        "configure color conversion",
                    )?;
                    if (*self.rgb_frame).width != frame.width
                        || (*self.rgb_frame).height != frame.height
                        || (*self.rgb_frame).format != output_format as i32
                    {
                        av::av_frame_unref(self.rgb_frame);
                        (*self.rgb_frame).width = frame.width;
                        (*self.rgb_frame).height = frame.height;
                        (*self.rgb_frame).format = output_format as i32;
                        check(
                            av::av_frame_get_buffer(self.rgb_frame, 32),
                            "allocate RGB frame",
                        )?;
                    }
                    let rows = av::sws_scale(
                        self.scale,
                        frame.data.as_ptr() as *const *const u8,
                        frame.linesize.as_ptr(),
                        0,
                        frame.height,
                        (*self.rgb_frame).data.as_ptr(),
                        (*self.rgb_frame).linesize.as_ptr(),
                    );
                    if rows != frame.height {
                        return Err(failure("color conversion failed"));
                    }
                    let row_bytes = width * 3 * if float_output { 2 } else { 1 };
                    for row in 0..height {
                        ptr::copy_nonoverlapping(
                            (*self.rgb_frame).data[0]
                                .add(row * (*self.rgb_frame).linesize[0] as usize),
                            pixels.as_mut_ptr().add(first * stride + row * row_bytes),
                            row_bytes,
                        );
                    }
                    for &position in &positions {
                        if position != first {
                            pixels.copy_within(
                                first * stride..(first + 1) * stride,
                                position * stride,
                            );
                        }
                        pts[position] = self.seconds(decoded_pts);
                        durations[position] = self.seconds(frame.duration);
                    }
                }
                break;
            }
        }
        pixels.truncate(total * if float_output { 2 } else { 1 });
        let pixels = if float_output {
            Pixels::F32(
                pixels
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| u16::from_le_bytes([p[0], p[1]]) as f32 / 65535.)
                    .collect(),
            )
        } else {
            Pixels::U8(pixels)
        };
        Ok(Video {
            pixels,
            pts,
            durations,
            width,
            height,
        })
    }

    pub fn audio(&mut self, rate: i32, channels: i32, stop: Option<f64>) -> Result<Audio> {
        if !self.audio {
            return Err(failure("cannot decode audio from video stream"));
        }
        if rate <= 0 || channels <= 0 || channels > 64 {
            return Err(Error(
                "sample_rate and num_channels must be positive (channels <= 64)".into(),
                true,
            ));
        }
        // Include codec preroll before time zero (e.g. AAC priming packets).
        if self
            .seek(self.begin_pts().min(0).saturating_sub(1))
            .is_err()
        {
            self.seek(0)?;
        }
        let mut resampler = Resampler(ptr::null_mut());
        let mut planes = vec![Vec::<f32>::new(); channels as usize];
        let mut first_pts = None;
        while self.next()? {
            let pts = self.seconds(self.frame_pts()?);
            first_pts.get_or_insert(pts);
            unsafe {
                let frame = &*self.frame;
                if resampler.0.is_null() {
                    let mut layout = std::mem::zeroed::<av::AVChannelLayout>();
                    av::av_channel_layout_default(&mut layout, channels);
                    let input_format: av::AVSampleFormat = std::mem::transmute(frame.format);
                    let code = av::swr_alloc_set_opts2(
                        &mut resampler.0,
                        &layout,
                        av::AVSampleFormat::AV_SAMPLE_FMT_FLTP,
                        rate,
                        &frame.ch_layout as *const _ as *mut _,
                        input_format,
                        frame.sample_rate,
                        0,
                        ptr::null_mut(),
                    );
                    av::av_channel_layout_uninit(&mut layout);
                    check(code, "configure audio resampling")?;
                    check(av::swr_init(resampler.0), "initialize audio resampling")?;
                }
                resampler.convert(
                    &mut planes,
                    frame.extended_data as *mut *const u8,
                    frame.nb_samples,
                )?;
            }
            if stop.is_some_and(|end| {
                first_pts.unwrap_or(0.) + planes[0].len() as f64 / rate as f64 >= end
            }) {
                break;
            }
        }
        if !resampler.0.is_null() {
            unsafe {
                resampler.convert(&mut planes, ptr::null_mut(), 0)?;
            }
        }
        let samples = planes[0].len();
        Ok(Audio {
            data: planes.into_iter().flatten().collect(),
            samples,
            pts: first_pts.unwrap_or(self.seconds(self.begin_pts())),
        })
    }
}

struct Resampler(*mut av::SwrContext);
impl Drop for Resampler {
    fn drop(&mut self) {
        unsafe {
            av::swr_free(&mut self.0);
        }
    }
}
impl Resampler {
    unsafe fn convert(
        &mut self,
        planes: &mut [Vec<f32>],
        input: *mut *const u8,
        samples: i32,
    ) -> Result<()> {
        let capacity = av::swr_get_out_samples(self.0, samples);
        check(capacity, "get audio output size")?;
        let mut scratch = vec![vec![0f32; capacity as usize + 32]; planes.len()];
        let mut pointers: Vec<_> = scratch
            .iter_mut()
            .map(|p| p.as_mut_ptr() as *mut u8)
            .collect();
        let count = av::swr_convert(self.0, pointers.as_mut_ptr(), capacity, input, samples);
        check(count, "resample audio")?;
        for (plane, output) in planes.iter_mut().zip(scratch) {
            plane.extend_from_slice(&output[..count as usize]);
        }
        Ok(())
    }
}

pub enum Pixels {
    U8(Vec<u8>),
    F32(Vec<f32>),
}
pub struct Video {
    pub pixels: Pixels,
    pub pts: Vec<f64>,
    pub durations: Vec<f64>,
    pub width: usize,
    pub height: usize,
}
pub struct Audio {
    pub data: Vec<f32>,
    pub samples: usize,
    pub pts: f64,
}
