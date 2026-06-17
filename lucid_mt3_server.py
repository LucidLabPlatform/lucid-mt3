#!/usr/bin/env python3
"""
LUCID MT3 GPU server.

Runs on the Central Command machine (the only GPU box). Owns the demonstration
store and answers two kinds of request over MQTT, following the LUCID component
command/result convention:

  cmd/run         : run MT3 inference for a captured scene -> bottleneck pose + twists
  cmd/upload_demo : store a demonstration recorded on the robot

  subscribe : lucid/agents/<AGENT_ID>/components/mt3/cmd/run
              lucid/agents/<AGENT_ID>/components/mt3/cmd/upload_demo
  publish   : lucid/agents/<AGENT_ID>/components/mt3/evt/run/result
              lucid/agents/<AGENT_ID>/components/mt3/evt/upload_demo/result
  status    : lucid/agents/<AGENT_ID>/components/mt3/status   (retained)

The MT3 pipeline itself is unchanged research code: this server lays the request
inputs onto the fixed paths `deployment/deploy_mt3.py` reads from, runs that
script as a subprocess (identical to the robot's old `docker run`, minus the
container boundary), and ships back the two output arrays it writes. Inputs and
outputs cross the wire as raw `.npy` / `.png` bytes so they are byte-identical
to what the script reads and writes on disk -- no re-encoding, no channel-order
or dtype surprises.

Run inside the `thousand-tasks` image (CUDA torch + thousand_tasks installed):

  MQTT_HOST=<broker> MQTT_PORT=1883 \
  MQTT_USERNAME=<user> MQTT_PASSWORD=<pass> \
  LUCID_MT3_AGENT_ID=mt3 \
  python lucid_mt3_server.py

Standalone responder that intentionally speaks the SAME topic contract as a real
lucid component, so it can later be promoted into a proper `lucid-component-mt3`
with no change to the Kinova client.
"""

import os
import re
import json
import base64
import shutil
import logging
import threading
import subprocess
from collections import OrderedDict

import paho.mqtt.client as mqtt

LOG = logging.getLogger("lucid.mt3.server")

AGENT_ID = os.environ.get("LUCID_MT3_AGENT_ID", "mt3")
COMPONENT_ID = "mt3"
TOPIC_ROOT = os.environ.get(
    "LUCID_MT3_TOPIC",
    "lucid/agents/{}/components/{}".format(AGENT_ID, COMPONENT_ID),
)
RUN_CMD_TOPIC = "{}/cmd/run".format(TOPIC_ROOT)
RUN_EVT_TOPIC = "{}/evt/run/result".format(TOPIC_ROOT)
UPLOAD_CMD_TOPIC = "{}/cmd/upload_demo".format(TOPIC_ROOT)
UPLOAD_EVT_TOPIC = "{}/evt/upload_demo/result".format(TOPIC_ROOT)
STATUS_TOPIC = "{}/status".format(TOPIC_ROOT)

MQTT_HOST = os.environ["MQTT_HOST"]
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_USERNAME = os.environ.get("MQTT_USERNAME")
MQTT_PASSWORD = os.environ.get("MQTT_PASSWORD")

# Paths inside the thousand-tasks image. `deploy_mt3.py` reads/writes these by
# convention (assets/ is relative to the thousand_tasks package root).
WORKSPACE = os.environ.get("LUCID_MT3_WORKSPACE", "/workspace")
ASSETS_DIR = os.path.join(WORKSPACE, "assets")
INFERENCE_DIR = os.path.join(ASSETS_DIR, "inference_example")
DEMOS_DIR = os.path.join(ASSETS_DIR, "demonstrations")
SAVED_DIR = os.path.join(WORKSPACE, "saved_data")
T_WC_PATH = os.path.join(ASSETS_DIR, "T_WC_head.npy")
DEPLOY_SCRIPT = "deployment/deploy_mt3.py"
SUBPROCESS_TIMEOUT = float(os.environ.get("LUCID_MT3_RUN_TIMEOUT", "280"))

# The nine files a complete demonstration must contain (MT3 inference subset;
# arm_joint_positions.npy / metadata.pkl are for BC training and not required).
REQUIRED_DEMO_FILES = (
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

# Inputs deploy_mt3.py reads from assets/inference_example/ (T_WC is separate).
RUN_INPUT_FILES = {
    "rgb_b64": "head_camera_ws_rgb.png",
    "depth_b64": "head_camera_ws_depth_to_rgb.png",
    "segmap_b64": "head_camera_ws_segmap.npy",
    "intrinsics_b64": "head_camera_rgb_intrinsic_matrix.npy",
}


def _make_client(client_id):
    # Works on both paho-mqtt 1.x and 2.x.
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=client_id)
    except (AttributeError, TypeError):
        return mqtt.Client(client_id=client_id)


def _b64_to_file(b64, path):
    with open(path, "wb") as fh:
        fh.write(base64.b64decode(b64))


def _file_to_b64(path):
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


class Mt3Server:
    def __init__(self):
        self._gpu_lock = threading.Lock()   # serialize inference (one GPU)
        self._fs_lock = threading.Lock()    # serialize demo-folder assignment
        self._seen = OrderedDict()          # request_id dedup
        self._seen_max = 256

        os.makedirs(INFERENCE_DIR, exist_ok=True)
        os.makedirs(DEMOS_DIR, exist_ok=True)
        os.makedirs(SAVED_DIR, exist_ok=True)

        self._client = _make_client("lucid-mt3-{}".format(os.getpid()))
        if MQTT_USERNAME:
            self._client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.will_set(
            STATUS_TOPIC, json.dumps({"state": "error"}), qos=1, retain=True
        )

    # -- MQTT plumbing -------------------------------------------------------

    def _is_duplicate(self, rid):
        if not rid:
            return False
        if rid in self._seen:
            return True
        self._seen[rid] = True
        if len(self._seen) > self._seen_max:
            self._seen.popitem(last=False)
        return False

    def _on_connect(self, client, userdata, flags, rc, *args):
        LOG.info("Connected to MQTT (rc=%s). Subscribing to run + upload_demo.", rc)
        client.subscribe(RUN_CMD_TOPIC, qos=1)
        client.subscribe(UPLOAD_CMD_TOPIC, qos=1)
        client.publish(
            STATUS_TOPIC, json.dumps({"state": "running"}), qos=1, retain=True
        )

    def _on_message(self, client, userdata, msg):
        try:
            req = json.loads(msg.payload.decode("utf-8"))
        except Exception as exc:
            LOG.warning("Bad request payload on %s: %s", msg.topic, exc)
            return
        rid = req.get("request_id", "")
        if self._is_duplicate(rid):
            LOG.info("Duplicate request_id %s ignored", rid)
            return
        if msg.topic == RUN_CMD_TOPIC:
            target, evt = self._handle_run, RUN_EVT_TOPIC
        elif msg.topic == UPLOAD_CMD_TOPIC:
            target, evt = self._handle_upload_demo, UPLOAD_EVT_TOPIC
        else:
            LOG.warning("Unexpected topic %s", msg.topic)
            return
        # Offload so the network loop stays responsive.
        threading.Thread(
            target=self._run_handler, args=(target, req, rid, evt), daemon=True
        ).start()

    def _run_handler(self, target, req, rid, evt_topic):
        try:
            result = target(req, rid)
        except Exception as exc:
            LOG.exception("Handler failed for request %s", rid)
            result = {"request_id": rid, "ok": False, "error": str(exc)}
        self._client.publish(evt_topic, json.dumps(result), qos=1)

    # -- cmd/run -------------------------------------------------------------

    def _handle_run(self, req, rid):
        task_name = req.get("task_name")
        if not task_name:
            return {"request_id": rid, "ok": False, "error": "missing task_name"}

        with self._gpu_lock:
            # Lay the request inputs onto the fixed paths deploy_mt3.py reads.
            if os.path.isdir(INFERENCE_DIR):
                shutil.rmtree(INFERENCE_DIR)
            os.makedirs(INFERENCE_DIR, exist_ok=True)
            for key, fname in RUN_INPUT_FILES.items():
                if key not in req:
                    return {"request_id": rid, "ok": False,
                            "error": "missing input '{}'".format(key)}
                _b64_to_file(req[key], os.path.join(INFERENCE_DIR, fname))
            if "t_wc_b64" not in req:
                return {"request_id": rid, "ok": False, "error": "missing t_wc_b64"}
            _b64_to_file(req["t_wc_b64"], T_WC_PATH)

            # Clear stale outputs so we never read a previous run's result.
            for fname in ("live_bottleneck_pose.npy", "end_effector_twists.npy"):
                fpath = os.path.join(SAVED_DIR, fname)
                if os.path.exists(fpath):
                    os.remove(fpath)

            env = dict(os.environ)
            env["MPLBACKEND"] = "Agg"   # headless matplotlib savefig
            # Ensure `import thousand_tasks` resolves regardless of how the base
            # image set things up (cwd alone is not enough -- sys.path[0] is the
            # script's own directory).
            env["PYTHONPATH"] = WORKSPACE + os.pathsep + env.get("PYTHONPATH", "")
            proc = subprocess.run(
                ["python", DEPLOY_SCRIPT, "--task_name", task_name],
                cwd=WORKSPACE,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=SUBPROCESS_TIMEOUT,
            )
            stdout = proc.stdout.decode("utf-8", "replace") if proc.stdout else ""

            if proc.returncode != 0:
                if "No demonstrations exist for skill" in stdout:
                    err = "no demonstrations for task_name '{}'".format(task_name)
                else:
                    err = "deploy_mt3 failed (rc={}): {}".format(
                        proc.returncode, stdout[-800:]
                    )
                LOG.warning("Run %s failed: %s", rid, err)
                return {"request_id": rid, "ok": False, "error": err}

            pose_path = os.path.join(SAVED_DIR, "live_bottleneck_pose.npy")
            twists_path = os.path.join(SAVED_DIR, "end_effector_twists.npy")
            if not (os.path.exists(pose_path) and os.path.exists(twists_path)):
                return {"request_id": rid, "ok": False,
                        "error": "deploy_mt3 produced no outputs: {}".format(stdout[-800:])}

            pose_b64 = _file_to_b64(pose_path)
            twists_b64 = _file_to_b64(twists_path)

        retrieved = None
        match = re.search(r"Retrieved:\s*(\S+)", stdout)
        if match:
            retrieved = "demonstrations/{}".format(match.group(1))

        LOG.info("Run %s ok: retrieved=%s", rid, retrieved)
        return {
            "request_id": rid,
            "ok": True,
            "error": None,
            "bottleneck_pose_b64": pose_b64,
            "twists_b64": twists_b64,
            "retrieved_demo": retrieved,
        }

    # -- cmd/upload_demo -----------------------------------------------------

    def _handle_upload_demo(self, req, rid):
        task_name = req.get("task_name")
        files = req.get("files")
        if not task_name or not isinstance(files, dict):
            return {"request_id": rid, "ok": False,
                    "error": "missing task_name or files"}

        missing = [name for name in REQUIRED_DEMO_FILES if name not in files]
        if missing:
            return {"request_id": rid, "ok": False,
                    "error": "missing demo files: {}".format(missing)}

        with self._fs_lock:
            folder = self._next_demo_folder(task_name)
            final_dir = os.path.join(DEMOS_DIR, folder)
            tmp_dir = os.path.join(DEMOS_DIR, ".tmp_{}".format(rid))
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir)
            try:
                for name, b64 in files.items():
                    # Guard against path traversal in supplied file names.
                    if os.path.sep in name or name in ("", ".", ".."):
                        raise ValueError("illegal file name '{}'".format(name))
                    _b64_to_file(b64, os.path.join(tmp_dir, name))
                os.rename(tmp_dir, final_dir)   # atomic on the same filesystem
            except Exception:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise

        stored_at = "demonstrations/{}".format(folder)
        LOG.info("Stored demo %s", stored_at)
        return {"request_id": rid, "ok": True, "error": None, "stored_at": stored_at}

    def _next_demo_folder(self, task_name):
        """Return the next free `<task_name>_<NNN>` folder name.

        Existing bare `<task_name>` (pre-seeded) and `<task_name>_NNN` dirs are
        all considered; the new demo always gets the next 3-digit number. The
        retrieval parser strips a trailing 3-digit suffix to recover the
        skill+object, so this is the naming convention it expects.
        """
        pattern = re.compile(r"^{}_(\d{{3}})$".format(re.escape(task_name)))
        nums = []
        if os.path.isdir(DEMOS_DIR):
            for entry in os.listdir(DEMOS_DIR):
                m = pattern.match(entry)
                if m:
                    nums.append(int(m.group(1)))
        nxt = (max(nums) + 1) if nums else 1
        return "{}_{:03d}".format(task_name, nxt)

    def run(self):
        self._client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
        self._client.loop_forever()


def main():
    logging.basicConfig(
        level=os.environ.get("LUCID_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    Mt3Server().run()


if __name__ == "__main__":
    main()
