# LUCID MT3 server -- a thin MQTT layer on top of the proven `thousand-tasks`
# image (the same image the robot builds and runs today).
#
# The base image already contains the full MT3 stack: PyTorch 2.0.1 + CUDA 11.8,
# Open3D, torch-geometric, CLIP, the `thousand_tasks` package, the `.ckpt`
# weights, and `deployment/deploy_mt3.py` itself. We add only paho-mqtt and the
# server script -- no MT3 dependencies are duplicated here.
#
# Prerequisite (one-time, on the CC host) -- build the base image from an
# once-project checkout:
#   cd ~/once-project/1000_tasks/learning_thousand_tasks
#   docker build -t thousand-tasks .
#
# Then:  docker compose up -d --build
#
# Override the base image name with --build-arg BASE_IMAGE=<name:tag> if you
# tag it differently.

ARG BASE_IMAGE=thousand-tasks
FROM ${BASE_IMAGE}

# Install into the same interpreter that runs deploy_mt3.py.
RUN python -m pip install --no-cache-dir "paho-mqtt>=2.0,<3"

COPY lucid_mt3_server.py /app/lucid_mt3_server.py
WORKDIR /app

CMD ["python", "lucid_mt3_server.py"]
