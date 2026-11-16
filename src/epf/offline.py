"""macOS model workers: no network, no raw data, only controller-supplied causal inputs.

The controller/scorer is trusted. Workers have read access only to installed code, the
pinned Chronos cache and their own checkpoint directory. This is not a claim about
what was in the foundation model's original pretraining corpus.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import select
import struct
import subprocess
import sys
import tempfile

from .data import ROOT


def disable_controller_network():
    """No downloads in the trusted controller; worker networking is denied by the OS."""
    def deny(event, args):
        if event.startswith("socket."):
            raise PermissionError("offline controller: network operations are disabled")
    sys.addaudithook(deny)


def send(stream, obj):
    data = pickle.dumps(obj, protocol=5)
    stream.write(struct.pack("!Q", len(data)))
    stream.write(data)
    stream.flush()


def receive(stream, heartbeat=False):
    if heartbeat:
        while not select.select([stream], [], [], 30)[0]:
            (ROOT / "runs/HEARTBEAT").touch()
    header = stream.read(8)
    if len(header) != 8:
        raise RuntimeError("offline worker exited before replying; inspect its stderr")
    size = struct.unpack("!Q", header)[0]
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise RuntimeError("truncated offline worker reply")
        chunks.extend(chunk)
    return pickle.loads(chunks)


class ModelWorker:
    def __init__(self, label, threads=8):
        if sys.platform != "darwin":
            raise RuntimeError("offline runner currently requires macOS sandbox-exec")
        checkpoints = ROOT / "runs" / "ckpt"
        checkpoints.mkdir(parents=True, exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix=f"offline_{label}_", dir=checkpoints)).resolve()
        self.cache = Path.home() / ".cache/huggingface/hub/models--amazon--chronos-2"
        revision = "29ec3766d36d6f73f0696f85560a422f50e8498c"
        self.snapshot = self.cache / "snapshots" / revision
        if label == "chronos" and not (self.snapshot / "config.json").exists():
            raise RuntimeError("Chronos must already be cached; offline workers never download models")
        readable = ["/System", "/usr", "/bin", "/sbin", "/Library", "/opt/homebrew",
                    "/private/etc", "/private/var/db", "/dev", sys.base_prefix,
                    str(Path(sys.executable).resolve().parents[1]), str(ROOT / ".venv"),
                    str(ROOT / "src"), str(self.work)]
        if label == "chronos":
            readable.append(str(self.cache))
            readable.extend(str(p.resolve()) for p in self.snapshot.iterdir() if p.is_file())
        rules = " ".join(f"(subpath {json.dumps(p)})" for p in readable)
        worker = ROOT / "scripts/offline_worker.py"
        profile = ("(version 1) (allow default) (deny network*) (deny file-read-data) "
                   f'(allow file-read-data (literal "/") {rules} (literal {json.dumps(str(worker))})) '
                   "(deny file-write*) "
                   f"(allow file-write* (subpath {json.dumps(str(self.work))}) "
                   '(literal "/dev/null") (literal "/dev/tty"))')
        self.profile = self.work / "sandbox.sb"
        self.profile.write_text(profile)
        print(f"offline worker {label}: checkpoints {self.work}", file=sys.stderr, flush=True)
        env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin", "HOME": str(self.work),
               "TMPDIR": str(self.work), "PYTHONPATH": str(ROOT / "src"),
               "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
               "TABPFN_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
               **{k: str(threads) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
        self.process = subprocess.Popen(["/usr/bin/sandbox-exec", "-f", str(self.profile),
                                         sys.executable, "-u", str(worker)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        cwd=self.work, env=env)
        try:
            # Probe a real synthetic file: ENOENT for an absent market-data file
            # says nothing about sandbox denial in a fresh checkout.
            with tempfile.TemporaryDirectory(prefix="epf_denied_probe_") as probe_dir:
                canary = Path(probe_dir) / "unreleased.txt"
                canary.write_text("synthetic isolation canary")
                self.call("probe", forbidden=str(canary))
        except Exception:
            self.close()
            raise

    def call(self, command, **payload):
        send(self.process.stdin, {"command": command, **payload})
        result = receive(self.process.stdout, heartbeat=True)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["value"]

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
        self.process.stdout.close()
        # Keep checkpoint directories, including failed fits, for inspection/resume.

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
