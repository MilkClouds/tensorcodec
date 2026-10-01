import numpy as np
import pytest

from tensorcodec.decoders import VideoDecoder
from tensorcodec.transforms import CenterCrop, RandomCrop, Resize
from tests.utils import as_numpy, index_input, run_ffmpeg


@pytest.fixture(scope="session")
def scene(tmp_path_factory):
    """Detailed 4:2:0 content, so resampling differences show."""
    path = tmp_path_factory.mktemp("transforms") / "scene.mp4"
    run_ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=10:duration=0.5",
        "-c:v", "libx264", "-threads", "1", "-crf", "0", "-g", "2", path,
    )  # fmt: skip
    return path


def _frames(path, transforms=(), **kwargs):
    with VideoDecoder(path, dimension_order="NHWC", transforms=transforms, **kwargs) as decoder:
        return decoder.get_frames_at([4, 0, 4]).data


def _ffmpeg_resized(path, height, width, pixel_format="rgb24"):
    """FFmpeg's own pipeline: RGB conversion, then a bilinear resize in RGB."""
    import subprocess

    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-threads", "1", "-vf",
         f"format={pixel_format},scale={width}:{height}:flags=bilinear", "-f", "rawvideo", "-pix_fmt", pixel_format,
         "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    dtype = np.uint8 if pixel_format == "rgb24" else "<u2"
    return np.frombuffer(out, dtype=dtype).reshape(-1, height, width, 3)


@pytest.mark.parametrize("size", [(32, 48), (40, 72), (128, 160)])  # down by 2, uneven, up
def test_resize_is_bilinear_in_rgb(scene, size):
    frames = _frames(scene, [Resize(size)])
    assert frames.shape == (3, *size, 3)
    expected = _ffmpeg_resized(scene, *size)[[4, 0, 4]]
    np.testing.assert_allclose(frames, expected, atol=1, rtol=0)


def test_crops_select_exact_pixels(scene):
    full = _frames(scene)
    np.testing.assert_array_equal(_frames(scene, [CenterCrop((31, 50))]), full[:, 16:47, 23:73])
    np.random.seed(3)
    top, left = np.random.randint(0, 64 - 20 + 1), np.random.randint(0, 96 - 30 + 1)
    np.random.seed(3)
    np.testing.assert_array_equal(_frames(scene, [RandomCrop((20, 30))]), full[:, top : top + 20, left : left + 30])


def test_a_pipeline_applies_in_order(scene):
    resized = _frames(scene, [Resize((32, 48))])
    np.testing.assert_array_equal(_frames(scene, [Resize((32, 48)), CenterCrop((16, 16))]), resized[:, 8:24, 16:32])
    cropped = _frames(scene, [CenterCrop((32, 48)), Resize((16, 24))])
    assert cropped.shape == (3, 16, 24, 3)
    reference = _frames(scene, [CenterCrop((32, 48))])
    assert np.abs(cropped.astype(int) - reference[:, ::2, ::2]).mean() < 12  # the same picture, smaller


@pytest.mark.parametrize("angle", [90, 180, -90])
def test_transforms_apply_after_display_rotation(scene, tmp_path, angle):
    path = tmp_path / "rotated.mp4"
    run_ffmpeg("-display_rotation", angle, "-i", scene, "-c", "copy", path)
    full = _frames(path)  # display orientation
    height, width = full.shape[1:3]
    np.random.seed(0)
    crops = [RandomCrop((height - 7, width - 11)) for _ in range(3)]
    for crop in crops:
        np.random.seed(1)
        top, left = np.random.randint(0, 8), np.random.randint(0, 12)
        np.random.seed(1)
        np.testing.assert_array_equal(_frames(path, [crop]), full[:, top : top + height - 7, left : left + width - 11])
    resized = _frames(path, [Resize((height // 2, width // 3))])
    assert resized.shape == (3, height // 2, width // 3, 3)
    upright = _frames(scene, [Resize((width // 3, height // 2) if angle % 180 else (height // 2, width // 3))])
    np.testing.assert_array_equal(resized, np.rot90(upright, angle // 90, axes=(1, 2)))


def test_high_depth_output_resizes_at_depth(scene):
    frames = _frames(scene, [Resize((32, 48))], output_dtype=np.uint16)
    assert frames.dtype == np.uint16 and frames.shape == (3, 32, 48, 3)
    expected = _ffmpeg_resized(scene, 32, 48, "rgb48le")[[4, 0, 4]]
    np.testing.assert_allclose(frames, expected, atol=1, rtol=0)
    floats = _frames(scene, [Resize((32, 48))], output_dtype=np.float32)
    np.testing.assert_allclose(floats, frames / np.float32(65535), atol=1e-7, rtol=0)


def test_timestamp_mode_and_empty_batches_take_the_output_size(scene):
    with VideoDecoder(scene, seek_mode="timestamp", dimension_order="NHWC", transforms=[Resize((16, 24))]) as d:
        assert d.get_frames_played_at([0.4, 0.0]).data.shape == (2, 16, 24, 3)
    with VideoDecoder(scene, transforms=[CenterCrop((10, 12))]) as d:
        assert d.get_frames_at([]).data.shape == (0, 3, 10, 12)


def test_invalid_transforms_are_refused(scene):
    for transforms in ([CenterCrop((65, 10))], [Resize((10, 10)), RandomCrop((11, 5))]):
        with pytest.raises(ValueError, match="exceeds"):
            VideoDecoder(scene, transforms=transforms)
    for size in [(0, 4), (4,), "ab", (2.0, 3)]:
        with pytest.raises(ValueError, match="size"):
            Resize(size)
    with pytest.raises(ValueError, match="RGB output"):
        VideoDecoder(scene, output_format="native", transforms=[Resize((8, 8))])


def test_torchcodec_and_torchvision_transforms_convert(scene):
    def counterpart(module, name, **fields):
        cls = type(name, (), {"__module__": module})
        obj = cls()
        obj.__dict__.update(fields)
        return obj

    expected = _frames(scene, [Resize((32, 48)), CenterCrop((16, 16))])
    tc = [counterpart("torchcodec.transforms._decoder_transforms", "Resize", size=[32, 48]),
          counterpart("torchcodec.transforms._decoder_transforms", "CenterCrop", size=[16, 16])]  # fmt: skip
    np.testing.assert_array_equal(_frames(scene, tc), expected)
    tv = counterpart("torchvision.transforms.v2._geometry", "Resize", size=[32, 48], interpolation="bilinear",
                     antialias=True)  # fmt: skip
    np.testing.assert_array_equal(_frames(scene, [tv]), _frames(scene, [Resize((32, 48))]))
    tv.interpolation = "nearest"
    with pytest.raises(ValueError, match="bilinear"):
        VideoDecoder(scene, transforms=[tv])


@pytest.mark.parametrize("width", [96, 64])  # torchcodec: swscale at multiples of 32, filtergraph otherwise
def test_transforms_match_torchcodec(oracle, scene, tmp_path, width):
    import torchcodec.transforms as tct

    path = scene
    if width != 96:
        path = tmp_path / "narrow.mp4"
        run_ffmpeg("-i", scene, "-vf", f"crop={width}:64", "-c:v", "libx264", "-crf", "0", path)
    for ours, theirs in [
        ([Resize((32, 48))], [tct.Resize((32, 48))]),
        ([Resize((40, 70))], [tct.Resize((40, 70))]),
        ([CenterCrop((30, 40))], [tct.CenterCrop((30, 40))]),
        ([CenterCrop((40, 40)), Resize((20, 24))], [tct.CenterCrop((40, 40)), tct.Resize((20, 24))]),
    ]:
        actual = VideoDecoder(path, transforms=ours).get_frames_at([4, 0, 4])
        expected = oracle.VideoDecoder(path, transforms=theirs).get_frames_at(index_input(oracle, [4, 0, 4]))
        assert actual.data.shape == tuple(expected.data.shape)
        np.testing.assert_allclose(actual.data, as_numpy(expected.data), atol=1, rtol=0)
