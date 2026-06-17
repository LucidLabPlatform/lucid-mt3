#!/usr/bin/env python3
"""
LUCID MT3 client.

Runs on the Kinova/ROS Python 3.8 machine. Talks to the `mt3` component on
Central Command over MQTT, using the LUCID component command/result convention.
Two operations:

    from lucid_mt3_client import Mt3Client

    mt3 = Mt3Client.from_env()

    # Run inference for a captured scene (assets the capture step just wrote):
    pose, twists = mt3.run(
        task_name="pick_up_bottle",
        inference_dir="/.../assets/inference_example",
        t_wc_path="/.../assets/T_WC_head.npy",
    )

    # Upload a recorded demonstration directory (the 9 MT3 files inside it):
    mt3.upload_demo(task_name="pick_up_bottle", demo_dir="/.../demo_0000")

Inputs and outputs are shipped as raw file bytes, so what the server feeds to
`deploy_mt3.py` is byte-identical to what this machine wrote on disk.

Environment variables used by `from_env`:
    MQTT_HOST        (required)
    MQTT_PORT        (default 1883)
    MQTT_USERNAME    (optional)
    MQTT_PASSWORD    (optional)
    LUCID_MT3_TOPIC  (default lucid/agents/mt3/components/mt3)

Python 3.8 compatible. Requires: paho-mqtt, numpy.
"""

import os
import io
import json
import time
import uuid
import base64
import threading

import numpy as np

try:
    import paho.mqtt.client as mqtt
except ImportError as exc:  # pragma: no cover
    raise ImportError("Mt3Client requires paho-mqtt: pip install paho-mqtt") from exc


# Inputs deploy_mt3.py reads from assets/inference_example/ -> request keys.
_RUN_INPUT_FILES = (
    ("rgb_b64", "head_camera_ws_rgb.png"),
    ("depth_b64", "head_camera_ws_depth_to_rgb.png"),
    ("segmap_b64", "head_camera_ws_segmap.npy"),
    ("intrinsics_b64", "head_camera_rgb_intrinsic_matrix.npy"),
)

# Files that must be present to upload a complete demonstration.
_REQUIRED_DEMO_FILES = (
    "task_name.txt",
    "demo_eef_twists.npy",
    "eef_poses.npy",
    "bottleneck_pose.npy",
    "timestamps.npy",
    "head_camera_ws_rgb.png",
    "head_camera_ws_depth_to_rgb.png",
    "head_camera_ws_segmap.npy",
    "head_camera_rgb_intrinsic_matrix.npy",
)


def _make_client(client_id):
    # Works on both paho-mqtt 1.x and 2.x.
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=client_id)
    except (AttributeError, TypeError):
        return mqtt.Client(client_id=client_id)


def _file_b64(path):
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _b64_npy(payload):
    return np.load(io.BytesIO(base64.b64decode(payload)), allow_pickle=False)


class Mt3Error(RuntimeError):
    """Raised when the server reports an error or a request times out."""


class Mt3Client:
    """Remote MT3 over MQTT. One persistent connection, blocking per call."""

    def __init__(
        self,
        host,
        port=1883,
        username=None,
        password=None,
        topic_root="lucid/agents/mt3/components/mt3",
        timeout=300.0,
        client_id=None,
    ):
        self.run_cmd_topic = "{}/cmd/run".format(topic_root)
        self.run_evt_topic = "{}/evt/run/result".format(topic_root)
        self.upload_cmd_topic = "{}/cmd/upload_demo".format(topic_root)
        self.upload_evt_topic = "{}/evt/upload_demo/result".format(topic_root)
        self.timeout = float(timeout)

        self._responses = {}
        self._lock = threading.Lock()
        self._event = threading.Event()

        cid = client_id or "lucid-mt3-client-{}".format(os.getpid())
        self._mqtt = _make_client(cid)
        if username:
            self._mqtt.username_pw_set(username, password)
        self._mqtt.on_message = self._on_message
        self._mqtt.connect(host, int(port), keepalive=60)
        self._mqtt.subscribe(self.run_evt_topic, qos=1)
        self._mqtt.subscribe(self.upload_evt_topic, qos=1)
        self._mqtt.loop_start()

    @classmethod
    def from_env(cls, **overrides):
        kwargs = dict(
            host=os.environ["MQTT_HOST"],
            port=int(os.environ.get("MQTT_PORT", "1883")),
            username=os.environ.get("MQTT_USERNAME"),
            password=os.environ.get("MQTT_PASSWORD"),
            topic_root=os.environ.get(
                "LUCID_MT3_TOPIC", "lucid/agents/mt3/components/mt3"
            ),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    # -- public API ----------------------------------------------------------

    def run(self, task_name, inference_dir, t_wc_path):
        """Run MT3 inference for a captured scene.

        Reads the four `inference_example` files + T_WC, ships their raw bytes,
        and returns (bottleneck_pose (4,4), twists (T,7)) numpy arrays.
        """
        payload = {"request_id": str(uuid.uuid4()), "task_name": task_name}
        for key, fname in _RUN_INPUT_FILES:
            payload[key] = _file_b64(os.path.join(inference_dir, fname))
        payload["t_wc_b64"] = _file_b64(t_wc_path)

        data = self._request(self.run_cmd_topic, payload)
        pose = _b64_npy(data["bottleneck_pose_b64"])
        twists = _b64_npy(data["twists_b64"])
        return _normalize_pose(pose), _normalize_twists(twists)

    def upload_demo(self, task_name, demo_dir):
        """Ship a recorded demonstration directory to CC.

        Reads the 9 required files from demo_dir and uploads them. The server
        stores the demo as a new `<task_name>_<NNN>` folder. Returns the server
        response dict (incl. `stored_at`). Raises Mt3Error on failure.
        """
        missing = [
            name for name in _REQUIRED_DEMO_FILES
            if not os.path.isfile(os.path.join(demo_dir, name))
        ]
        if missing:
            raise Mt3Error("demo_dir {} missing files: {}".format(demo_dir, missing))

        files = {
            name: _file_b64(os.path.join(demo_dir, name))
            for name in _REQUIRED_DEMO_FILES
        }
        payload = {
            "request_id": str(uuid.uuid4()),
            "task_name": task_name,
            "files": files,
        }
        return self._request(self.upload_cmd_topic, payload)

    def close(self):
        try:
            self._mqtt.loop_stop()
            self._mqtt.disconnect()
        except Exception:
            pass

    # -- internals -----------------------------------------------------------

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            return
        rid = data.get("request_id")
        if rid is None:
            return
        with self._lock:
            self._responses[rid] = data
        self._event.set()

    def _request(self, cmd_topic, payload):
        request_id = payload["request_id"]
        self._mqtt.publish(cmd_topic, json.dumps(payload), qos=1)
        data = self._wait_for(request_id)
        if data is None:
            raise Mt3Error("MT3 request {} timed out after {}s".format(
                request_id, self.timeout))
        if not data.get("ok"):
            raise Mt3Error("MT3 server error: {}".format(data.get("error")))
        return data

    def _wait_for(self, request_id):
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            with self._lock:
                data = self._responses.pop(request_id, None)
            if data is not None:
                return data
            self._event.wait(timeout=0.5)
            self._event.clear()
        return None


def _normalize_pose(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape == (1, 4, 4):
        pose = pose[0]
    if pose.shape != (4, 4):
        raise Mt3Error("bottleneck_pose has shape {}, expected (4, 4)".format(pose.shape))
    return pose


def _normalize_twists(twists):
    twists = np.asarray(twists, dtype=np.float64)
    if twists.ndim == 1:
        twists = twists.reshape(1, -1)
    if twists.ndim != 2 or twists.shape[1] < 6:
        raise Mt3Error(
            "twists must have shape (T, 6+) with [vx, vy, vz, wx, wy, wz, ...]"
        )
    return twists
