# lucid-mt3

GPU-accelerated MT3 inference service for the LUCID stack. Hosts the
retrieval-based manipulation pipeline (`thousand_tasks` / `deploy_mt3.py`) on
Central Command, serving MT3 inference and demonstration uploads over MQTT.

Mirrors the `lucid-langsam` pattern: standalone Docker container, EMQX broker on
the host network, identity provisioned through `lucid-auth`. It runs the proven
`deploy_mt3.py` as a subprocess inside the `thousand-tasks` image — a thin shim,
not a reimplementation.

## What it does

| Command           | Direction       | Purpose                                              |
|-------------------|-----------------|------------------------------------------------------|
| `cmd/run`         | client → server | Run MT3 on a live scene, return bottleneck + twists. |
| `cmd/upload_demo` | client → server | Ship a freshly recorded demonstration up to CC.      |

Demonstrations are stored on a host bind-mount (`/var/lib/lucid-mt3/demonstrations`),
survive container rebuilds, and are picked up by hierarchical retrieval at the
next inference request — no server restart needed.

## Topics

```
lucid/agents/mt3/components/mt3/cmd/run
lucid/agents/mt3/components/mt3/evt/run/result
lucid/agents/mt3/components/mt3/cmd/upload_demo
lucid/agents/mt3/components/mt3/evt/upload_demo/result
lucid/agents/mt3/components/mt3/status        (retained)
```

### `cmd/run` payload

```json
{
  "request_id":     "<uuid>",
  "task_name":      "pick_up_bottle",
  "rgb_b64":        "<base64 of head_camera_ws_rgb.png>",
  "depth_b64":      "<base64 of head_camera_ws_depth_to_rgb.png, 16-bit>",
  "segmap_b64":     "<base64 of head_camera_ws_segmap.npy>",
  "intrinsics_b64": "<base64 of head_camera_rgb_intrinsic_matrix.npy>",
  "t_wc_b64":       "<base64 of T_WC_head.npy>"
}
```

All inputs are the **raw file bytes** the capture step already wrote — shipped
verbatim so the server's inputs are byte-identical to the script's on-disk
inputs. Result:

```json
{
  "request_id": "<uuid>", "ok": true, "error": null,
  "bottleneck_pose_b64": "<base64 .npy (4,4)>",
  "twists_b64":          "<base64 .npy (T,7)>",
  "retrieved_demo":      "demonstrations/pick_up_bottle_001"
}
```

### `cmd/upload_demo` payload

```json
{
  "request_id": "<uuid>",
  "task_name":  "pick_up_bottle",
  "files": {
    "task_name.txt":                        "<base64>",
    "demo_eef_twists.npy":                  "<base64>",
    "eef_poses.npy":                        "<base64>",
    "bottleneck_pose.npy":                  "<base64>",
    "timestamps.npy":                       "<base64>",
    "head_camera_ws_rgb.png":               "<base64>",
    "head_camera_ws_depth_to_rgb.png":      "<base64>",
    "head_camera_ws_segmap.npy":            "<base64>",
    "head_camera_rgb_intrinsic_matrix.npy": "<base64>"
  }
}
```

`task_name` must be a structured skill string (`pick_up_bottle`, …) — see
**Demo store** below. The server assigns the folder name and never overwrites:

```json
{ "request_id": "<uuid>", "ok": true, "error": null,
  "stored_at": "demonstrations/pick_up_bottle_001" }
```

## Setup (one-time, on the CC host)

```bash
# 1. Get the base-image source (carries the MT3 stack + weights). once-project
#    is private, so on a host without git credentials for it, transfer the tree
#    out of band instead of cloning, e.g. from a machine that has it:
#      rsync -a <once-project>/1000_tasks/learning_thousand_tasks/ \
#            failsafe@<cc-host>:~/once-project/1000_tasks/learning_thousand_tasks/
#    (or `git clone https://github.com/IERoboticsAILab/once-project ~/once-project`
#     on the gpu-accelerated branch if you do have access.)
cd ~/once-project/1000_tasks/learning_thousand_tasks
docker build -t thousand-tasks .          # heavy; re-run only on algo/weights change

# 2. Provision MQTT credentials for the server identity (default `agent` role).
cd ~/lucid-central-command
docker compose exec auth python manage.py add-agent mt3

# 3. Configure. The demo store host path defaults to /var/lib/lucid-mt3/
#    demonstrations; on a host without passwordless sudo, point it at a
#    home-owned dir via MT3_DEMOS_HOST in .env.
cd lucid-mt3
cp .env.example .env          # set MT3_MQTT_PASSWORD (from step 2); optionally MT3_DEMOS_HOST

# 4. Pre-seed the existing demos into that store (already in the correct flat
#    layout, with geometry_encoding.npy cached).
DEMOS="${MT3_DEMOS_HOST:-/var/lib/lucid-mt3/demonstrations}"
mkdir -p "$DEMOS"            # prefix with sudo if it's a root-owned path
cp -a ~/once-project/1000_tasks/learning_thousand_tasks/assets/demonstrations/. "$DEMOS"/

# 5. Build + start.
docker compose up -d --build
```

Or just run `./deploy-cc.sh` from the `lucid-mt3` dir, which does steps 1, 4, 5
and verifies (it reads `MT3_DEMOS_HOST` and skips cloning if the source is
already present).

Verify:
```bash
docker logs --tail 20 lucid-mt3
# Expect: "Connected to MQTT ... Subscribing to run + upload_demo."
```

## Demo store — naming is load-bearing

Demos live flat in `/var/lib/lucid-mt3/demonstrations/`, one folder per demo:
`<task_name>_<NNN>/` (plus bare pre-seeded `<task_name>/`). Retrieval enumerates
with `glob('*')` and parses the **folder name** for skill+object, stripping a
trailing 3-digit suffix. So:

- `task_name` must be a structured skill string the parser understands
  (`pick_up_bottle`, `close_microwave_door`, …) — not a free-text prompt, not a
  timestamp. Other names are silently invisible to retrieval.
- The server assigns `<task_name>_<NNN>` (next free number); uploading the same
  task again adds `_002`, `_003`, … rather than overwriting.

## Operations

- **Add a demo:** record on the robot, run the collector with `--upload` (see
  `once-project/demo-scripts/`). Server stores `<task_name>_<NNN>`, retrieval
  picks it up next `cmd/run`.
- **Inspect the store:** `ls /var/lib/lucid-mt3/demonstrations/`.
- **Reset a task / all demos:** delete dirs under
  `/var/lib/lucid-mt3/demonstrations/` (destructive).
- **Update the algorithm or weights:** rebuild the `thousand-tasks` base image
  from a fresh once-project checkout, then `docker compose up -d --build`.

## Limits

- One inference in flight (GPU serialized via a process lock); concurrent
  `cmd/run` requests queue. MT3 is one-shot per task, so this is fine.
- Each `cmd/run` reloads the weights (a few seconds) — negligible against ICP
  (~3 s) + the human gate + the arm replay.
- EMQX packet cap is 4 MB; a demo upload (~1–3 MB) and an inference request
  (~1–2 MB) both fit.

## Files

- `lucid_mt3_server.py` — MQTT responder; subprocesses `deploy_mt3.py`.
- `Dockerfile` — `FROM thousand-tasks` + paho-mqtt + the server.
- `compose.yaml` — standalone, `network_mode: host`, NVIDIA GPU reservation,
  host bind-mount for demos.
- `.env.example` — required env vars (broker + creds).
