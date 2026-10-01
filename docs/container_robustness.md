# TensorCodec container and seek behavior

TensorCodec uses FFmpeg `libavformat` for demuxing and `libavcodec` for decoding
through its Rust/PyO3 boundary. Available codecs depend on the FFmpeg build.
Container support and seek behavior depend on the input's timestamps, index and
keyframe flags; a file extension alone does not establish correctness or speed.

## Tested cases

The [contract fixtures](../tests/conftest.py) generate known frame identities and
timestamps, checked independently with ffprobe and against TorchCodec 0.17.0.

| Container / codec | Case | Checks |
| --- | --- | --- |
| Matroska / FFV1 | CFR | Index and playback selection, frame identity, timing |
| MP4 / H.264 | VFR | Nonuniform PTS and durations, time-range selection |
| MP4 / H.264 | B-frames | Presentation ordering and seeking |
| Matroska / FFV1 | Nonzero start PTS | Offset timestamps and bounds |
| Matroska / FFV1 + PCM | Multiple streams | Container stream index selection |
| MP4 / H.264 | SDR color metadata | RGB conversion and float32 precision |

These fixtures establish the [supported playback contract](compatibility.md).
They are not an exhaustive matrix of every FFmpeg codec/container combination.

## Seeking

Exact mode scans packet timestamps and keyframe information when opening a
decoder, then seeks and decodes to the mapped PTS. Opening large inputs can add
I/O and startup cost; reuse a decoder when the workload permits it.

Approximate mode uses header FPS for index/time mapping. Use exact mode for VFR
frame selection. Detailed range and timing rules are in
[playback semantics](playback_semantics.md).

Seeking trusts container keyframe flags. Incorrect flags can produce damaged
frames even if a decoder returns the requested timestamp. Repair the input or
provide corrected frame mappings; TensorCodec does not guarantee automatic
recovery from incorrect container metadata.

## Measuring another workload

Use the [benchmark guide](../benchmarks/README.md) to generate codec/container
inputs and measure them with the current implementation. Report decoder-open
costs, seek mode, GOP/B-frame settings, cache conditions and partial timeouts.
Check decoded pixels and timestamps before drawing performance conclusions.
