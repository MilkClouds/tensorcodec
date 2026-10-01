# Video fidelity priorities

This ranking prioritizes correct pixels and common camera inputs before new
module families. It is based on the decoder's current implementation, not a
measured distribution of user workloads.

| Priority | Gap and impact | Status in this change |
| --- | --- | --- |
| P0 | Display rotation: phone videos were rejected instead of decoded upright | Apply 90/180/270-degree display rotation; report output dimensions |
| P0 | High bit depth: `auto` silently selected 8-bit output for every input | Inspect FFmpeg pixel descriptors; select float32 above 8 bits; add uint16 RGB |
| P0 | PQ/HLG: HDR inputs were rejected despite an available high-precision conversion path | Decode transfer-encoded RGB; preserve stream color metadata |
| P1 | Missing depth/range metadata obscured precision and YUV interpretation | Add `bit_depth` and `color_range` |
| P1 | HDR-to-SDR display conversion | Follow-up: explicit tone mapping, transfer conversion and gamut mapping with reference fixtures |
| P1 | Raw source planes and alpha | Follow-up: a separate planar output contract with subsampling, strides, native code values and ownership |
| P1 | Audio range performance | Follow-up: seek with codec preroll while preserving sample-accurate timing |
| P2 | Expensive exact-open scans and unreliable container keyframe flags | Follow-up: reusable indexes and stronger container validation |
| P2 | Resize/crop transforms, arbitrary display matrices and dynamic dimensions | Follow-up: explicit output geometry and interpolation contracts |
| P2 | GPU decode, image decode, encoders and additional wheel platforms | Separate projects; expand after CPU fidelity is covered |

## Output contract

`VideoDecoder(..., output_dtype=...)` always returns three-channel RGB:

| Value | Array dtype | Values |
| --- | --- | --- |
| `np.uint8` (default) | uint8 | 0–255; quantizes high-depth inputs |
| `np.uint16` | uint16 | 0–65535 via FFmpeg RGB48 conversion |
| `np.float32` | float32 | RGB48 values divided by 65535 |
| `"auto"` | uint8 or float32 | float32 when the source component depth exceeds 8 bits |

The `auto` heuristic follows TorchCodec's pixel-depth test, including high-depth
SDR. Explicit uint16 is a TensorCodec extension. It preserves a high-precision
RGB conversion, **not original YUV samples or native 0–1023/0–4095 code values**.
Color conversion can round and clip values outside the destination RGB range.
Float output does not pass through uint8.

PQ (`smpte2084`) and HLG (`arib-std-b67`) inputs retain their transfer encoding
and primaries. Output is not linear light, SDR/sRGB, or tone mapped. Consumers
must use the color metadata for subsequent display conversion. Dynamic HDR
metadata, mastering-display metadata and Dolby Vision processing are not exposed.

`metadata.bit_depth` describes the source component depth; `color_range` is
FFmpeg's stream range name (`tv`, `pc`, or None). Other color fields and
`pixel_format` also describe the source, not the full-range RGB output.
The dtype is selected once from stream metadata. Per-frame format/metadata
changes are not a supported fidelity contract.

Rotation is counter-clockwise in degrees, matching FFmpeg's display matrix.
`metadata.width` and `height` describe the rotated output; pixel aspect ratio
continues to describe the encoded stream. Rotation applies to both layouts,
all dtypes, empty batches and every frame retrieval method. Returned rotated
arrays have positive strides and retain ownership after close. Non-right-angle,
reflected and singular matrices fail explicitly; arbitrary affine display
transforms remain unsupported.

## Verification

`tests/test_video_fidelity.py` generates lossless 8/10/12/16-bit ramps, PQ/HLG
fixtures and non-square rotated videos. It checks independent FFmpeg RGB output,
precision above 256 levels, dtype selection, metadata, timing, request ordering,
empty shapes, array lifetime and DLPack. Existing playback and pinned-reference
checks remain in the regular suite.

Reference semantics: [TorchCodec HDR example](https://github.com/meta-pytorch/torchcodec/blob/main/examples/decoding/hdr_decoding.py)
and [FFmpeg pixel/rotation helpers in TorchCodec](https://github.com/meta-pytorch/torchcodec/blob/main/src/torchcodec/_core/FFMPEGCommon.cpp).
