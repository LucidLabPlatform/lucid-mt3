# CLAUDE.md — lucid-mt3

> Part of the **LUCID** monorepo. Sibling of `lucid-langsam` — same shape
> (standalone GPU MQTT responder), different model.

## Purpose

GPU-accelerated MT3 inference service. Two MQTT commands:

- `cmd/upload_demo` — robot sends a freshly recorded demonstration; server
  stores it on a persistent host-mounted volume.
- `cmd/run` — robot sends a live scene; server runs the `thousand_tasks`
  retrieval + PointNet++ + ICP pipeline and returns the live bottleneck pose
  plus the demo's recorded twist sequence.

Pairs with the vendored client in
`once-project/demo-scripts/lucid_mt3_client.py`.

## How it runs the pipeline

The server does **not** reimplement MT3. It shells out to the proven
`deployment/deploy_mt3.py` inside the `thousand-tasks` image — the exact script
the robot ran via `docker run`, minus the container boundary. Per `cmd/run`,
under a GPU lock, it:

1. Writes the request inputs onto the fixed paths the script reads
   (`assets/inference_example/*`, `assets/T_WC_head.npy`).
2. Runs `python deployment/deploy_mt3.py --task_name <name>` (with
   `MPLBACKEND=Agg`, `PYTHONPATH=/workspace`).
3. Reads back `saved_data/{live_bottleneck_pose,end_effector_twists}.npy` and
   ships them as raw `.npy` bytes.

Inputs and outputs cross the wire as raw file bytes — byte-identical to what the
script reads/writes on disk, so there's no re-encoding or dtype/channel-order
risk. Single source of truth: the pipeline lives only in once-project; this is a
thin shim.

## Key files

- `lucid_mt3_server.py` — the entire server. paho-mqtt, one lock serializing
  GPU runs, a separate lock for demo-folder assignment. No torch/numpy imports.
- `Dockerfile` — `FROM thousand-tasks` + `pip install paho-mqtt` + copy server.
  The base image carries the whole MT3 stack and weights. Build the base once on
  the CC host from an once-project checkout (`docker build -t thousand-tasks .`).
- `compose.yaml` — standalone (not wired into the umbrella compose). Host
  network; host bind-mount `/var/lib/lucid-mt3/demonstrations`.
- `.env.example` — minimum required env vars.

## Topic contract (do not invent variants)

```
lucid/agents/mt3/components/mt3/cmd/run
lucid/agents/mt3/components/mt3/cmd/upload_demo
lucid/agents/mt3/components/mt3/evt/run/result
lucid/agents/mt3/components/mt3/evt/upload_demo/result
lucid/agents/mt3/components/mt3/status       (retained: "running" / "error")
```

`AGENT_ID` is `mt3` (override with `LUCID_MT3_AGENT_ID`).

## Demo store — naming is load-bearing

Demos live flat, one level, in `/workspace/assets/demonstrations/`:
`<task_name>_<NNN>/` (or a bare pre-seeded `<task_name>/`). The retrieval code
enumerates the store with a single `glob('*')` and parses the **folder name**
for skill+object, stripping a trailing 3-digit suffix
(`thousand_tasks/data/utils.py:remove_demo_number_from_task_folder_name`).

- `task_name` must be a structured skill string (`pick_up_bottle`,
  `close_microwave_door`, …) the parser understands — **not** a free-text
  prompt and **not** a timestamp. Arbitrary names are invisible to retrieval.
- The server assigns the folder name: it scans existing `<task_name>_NNN` dirs
  and writes the next free `_NNN`. Never overwrites, no client-supplied id.

## Conventions

- Structured logging only — `logging.getLogger("lucid.mt3.server")`.
- Request payloads are JSON; binary blobs (`*_b64`, `files`) are base64 raw
  file bytes.
- One inference in flight (GPU lock); uploads run independently but serialize
  among themselves for folder-number assignment.

## Don'ts

- Don't reimplement the pipeline in-process — shell out to `deploy_mt3.py` so
  there's nothing to drift as the research code evolves.
- Don't use nested `<task>/<id>/` storage or timestamp folder names — they
  break retrieval (see above).
- Don't bump the MT3 stack here — it's owned by the `thousand-tasks` base image
  in once-project.

## See also

- `../lucid-langsam/` — the pattern this service mirrors.
- `.../once-project/1000_tasks/learning_thousand_tasks/deployment/deploy_mt3.py`
  — the pipeline this service hosts.
- `.../once-project/.../thousand_tasks/retrieval/` — retrieval + folder-name
  parsing (`hierarchical_retrieval.py`, `language_template_parser.py`).
- `/topics.txt` at the LUCID root for the MQTT topic schema.
