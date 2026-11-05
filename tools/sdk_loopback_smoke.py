"""Exercise the installed asMagic SDK on local TCP with no physical device.

Run: python tools/sdk_loopback_smoke.py --output recordings/sdk-loopback
This module can be imported without creating sockets or writing files.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from iphone_mujoco.recording import RecordingWriter, ReplaySource
from iphone_mujoco.streams import AsMagicSource


def run_loopback(output: Path) -> dict:
    try:
        import asmagic
        import zmq
        from asmagic.proto import joystick_msg_pb2, phone_msg_pb2
    except ImportError as error:
        raise ImportError("Install this project with its [live] extra before the SDK loopback smoke") from error
    output = Path(output).expanduser()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    context = zmq.Context()
    pose_publisher = context.socket(zmq.PUB)
    buttons_publisher = context.socket(zmq.PUB)
    for publisher in (pose_publisher, buttons_publisher):
        publisher.setsockopt(zmq.LINGER, 0)
    result = {"sdk_version": asmagic.__version__, "sdk_module": asmagic.__file__,
              "physical_iphone": False, "transport": "local TCP loopback"}
    try:
        pose_port = pose_publisher.bind_to_random_port("tcp://127.0.0.1")
        button_port = buttons_publisher.bind_to_random_port("tcp://127.0.0.1")
        with AsMagicSource("127.0.0.1", ar_port=pose_port, joystick_port=button_port,
                           stale_after_s=.5) as source:
            phone = phone_msg_pb2.Phone(timestamp=time.time(),
                local_pose=[.1, .2, .3, 0, 0, 0, 1], global_pose=[.1, .2, .3, 0, 0, 0, 1])
            buttons = joystick_msg_pb2.Joystick(timestamp=phone.timestamp, button_pressed=[False] * 4)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                pose_publisher.send(phone.SerializeToString())
                buttons_publisher.send(buttons.SerializeToString())
                time.sleep(.01)
                if source.latest and source.stats["joystick_frames"] and not source.latest.metadata["move_release_required"]:
                    break
            if not source.latest or source.latest.metadata["move_release_required"]:
                raise RuntimeError(f"Release-to-arm frames did not arrive: {source.stats}")
            buttons.button_pressed[:] = [False, True, False, False]
            buttons.face_buttons.square = True
            buttons.triggers.r2.value = .6
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                phone.timestamp = time.time()
                pose_publisher.send(phone.SerializeToString())
                buttons_publisher.send(buttons.SerializeToString())
                time.sleep(.01)
                if source.latest and not source.latest.clutch:
                    break
            sample = source.latest
            if sample is None or sample.clutch:
                raise RuntimeError(f"Move button did not enable control: {source.stats}")
            np.testing.assert_allclose(sample.pose[:3, 3], [-.3, -.1, .2], atol=1e-7)
            np.testing.assert_allclose(sample.gripper, .6, atol=1e-6)
            result["teleop"] = source.stats
        with AsMagicSource("127.0.0.1", mode="camera", ar_port=pose_port) as source:
            rgb = np.full((48, 64, 3), [40, 90, 180], dtype=np.uint8)
            jpeg = BytesIO()
            Image.fromarray(rgb).save(jpeg, format="JPEG")
            phone = phone_msg_pb2.Phone(timestamp=time.time(), source_timestamp=123.25, frame_id=42,
                local_pose=[.1, .2, .3, 0, 0, 0, 1], session_identifier="loopback-fixture",
                color_img=jpeg.getvalue(), rgb_width=64, rgb_height=48,
                camersIntrinsics=[60, 0, 0, 0, 61, 0, 31.5, 23.5, 1],
                depth_width=32, depth_height=24, depth_meters_per_unit=.0001,
                depth_img=np.full((24, 32), 10000, dtype="<u2").tobytes(),
                depth_confidence=np.full((24, 32), 2, dtype=np.uint8).tobytes(), rgb_codec=1)
            deadline = time.monotonic() + 3
            sample = None
            while time.monotonic() < deadline:
                pose_publisher.send(phone.SerializeToString())
                time.sleep(.01)
                sample = source.poll()
                if sample is not None and sample.rgbd is not None:
                    break
            if sample is None or sample.rgbd is None:
                raise RuntimeError(f"RGB-D fixture did not arrive: {source.stats}")
            np.testing.assert_allclose(sample.rgbd.depth_m, 1., atol=1e-6)
            if sample.rgbd.rgb.shape != (48, 64, 3):
                raise RuntimeError("Decoded RGB shape does not match the fixture")
            result["camera"] = {"frame_id": sample.frame_id,
                "sender_session_id": sample.metadata["sender_session_id"],
                "rgb_shape": list(sample.rgbd.rgb.shape), "depth_shape": list(sample.rgbd.depth_m.shape),
                "fx": sample.rgbd.calibration.fx, "depth_center_m": float(sample.rgbd.depth_m[12, 16])}
            with RecordingWriter(output / "episode") as writer:
                writer.write(sample)
            with ReplaySource(output / "episode") as replay:
                restored = replay.poll()
                if restored is None or restored.frame_id != 42:
                    raise RuntimeError("Replay did not preserve frame identity")
                np.testing.assert_array_equal(restored.rgbd.rgb, sample.rgbd.rgb)
                np.testing.assert_array_equal(restored.rgbd.depth_m, sample.rgbd.depth_m)
            result["recording_roundtrip"] = "episode"
    except zmq.ZMQError as error:
        raise RuntimeError(f"Local TCP loopback is unavailable: {error}") from error
    finally:
        pose_publisher.close()
        buttons_publisher.close()
        context.term()
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New output directory for result.json and episode/")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_loopback(args.output), indent=2))
        return 0
    except (OSError, RuntimeError, ValueError, ImportError, AssertionError) as error:
        print(f"Loopback smoke failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
