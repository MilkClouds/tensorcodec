# TensorCodec playback semantics

The supported behavior is specified in [compatibility.md](compatibility.md) and
executable tests in `tests/test_video_contract.py` and `tests/test_audio_contract.py`.
The reference is TorchCodec 0.17.0.

Exact video seeking scans presentation timestamps and preceding key-frame PTS.
Index requests use that map; time requests select the frame playing at the requested
time. The native decoder seeks to a key frame and decodes forward to the exact
PTS, restoring caller order and duplicates. It fails if the mapped frame cannot
be found rather than silently returning another frame.

Approximate seeking derives indices and time selection from header FPS. It does
not provide the content-derived VFR guarantees of exact mode.

For a time range, include the frame playing at its start and exclude a frame
starting at its stop. With `fps`, sample a uniform grid starting at the requested
start and return grid timestamps/durations, matching the reference implementation.
Packet duration and the next frame's PTS gap are distinct quantities.

Audio decoding preserves codec priming and encoder delay, then resamples with
FFmpeg. Range selection rounds offsets at the output sample rate. Current audio
queries replay decoding from the beginning; this trades speed for deterministic
range output and is a future optimization boundary.
