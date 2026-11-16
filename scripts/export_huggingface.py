"""Export the frozen paper checkpoint; never trains, reads outcomes, or uploads.

Run: .venv/bin/python scripts/export_huggingface.py
"""
from pathlib import Path
import gc
import hashlib
import json
import shutil
from importlib.metadata import version

import numpy as np
import requests
import torch
from chronos import Chronos2Pipeline

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "exports/chronos-2-de-lu-pregate"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    policy = json.loads((ROOT / "runs/paper/policy.json").read_text())
    adapter = ROOT / policy["adapter"]
    calibration = ROOT / "runs/readiness/calibration.json"
    assert digest(adapter / "adapter_model.safetensors") == policy["adapter_sha256"]
    assert digest(calibration) == policy["calibration_sha256"]
    config = json.loads((adapter / "adapter_config.json").read_text())
    assert Path(config["base_model_name_or_path"]).name == REVISION
    # Fetch licensing files only, pinned to the exact base revision.
    response = requests.get(f"https://huggingface.co/api/models/amazon/chronos-2/revision/{REVISION}", timeout=30)
    response.raise_for_status()
    siblings = {entry["rfilename"] for entry in response.json()["siblings"]}
    legal = {}
    for name in ("LICENSE", "NOTICE"):
        if name in siblings:
            r = requests.get(f"https://huggingface.co/amazon/chronos-2/resolve/{REVISION}/{name}", timeout=30)
            r.raise_for_status()
            legal[name] = r.text
    if response.json().get("cardData", {}).get("license") != "apache-2.0":
        raise RuntimeError("Unexpected base license; inspect before distributing weights")
    # The model repo declares Apache-2.0 in its card, but has no legal files.
    # Retain the license and attribution from its official source repository.
    upstream = requests.get("https://api.github.com/repos/amazon-science/chronos-forecasting/commits/main", timeout=30)
    upstream.raise_for_status()
    source_revision = upstream.json()["sha"]
    for name in ("LICENSE", "NOTICE"):
        if name not in legal:
            r = requests.get(f"https://raw.githubusercontent.com/amazon-science/chronos-forecasting/{source_revision}/{name}", timeout=30)
            r.raise_for_status()
            legal[name] = r.text
    DEST.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    # The pinned Chronos loader merges LoRA automatically. CPU export only.
    pipe = Chronos2Pipeline.from_pretrained(str(adapter), device_map="cpu", local_files_only=True)
    pipe.model.eval()
    x = np.arange(240, dtype=np.float32)
    item = {"target": 50 + 10 * np.sin(x / 24),
            "past_covariates": {"load_fc": 100 + x / 10},
            "future_covariates": {"load_fc": 100 + np.arange(240, 264, dtype=np.float32) / 10}}
    with torch.inference_mode():
        before = pipe.predict([item], prediction_length=24)[0].float().cpu().numpy()
    pipe.save_pretrained(DEST, safe_serialization=True)
    del pipe
    gc.collect()
    reloaded = Chronos2Pipeline.from_pretrained(str(DEST), device_map="cpu", local_files_only=True)
    reloaded.model.eval()
    with torch.inference_mode():
        after = reloaded.predict([item], prediction_length=24)[0].float().cpu().numpy()
    np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-5)
    if not np.isfinite(after).all():
        raise RuntimeError("Export produced non-finite predictions")
    del reloaded
    gc.collect()
    (DEST / "adapter").mkdir()
    shutil.copy2(adapter / "adapter_model.safetensors", DEST / "adapter/adapter_model.safetensors")
    config["base_model_name_or_path"] = "amazon/chronos-2"
    config["revision"] = REVISION
    (DEST / "adapter/adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy2(calibration, DEST / "calibration.json")
    for source, target in [("publish/model_card.md", "README.md"),
                           ("publish/inference.py", "inference.py")]:
        shutil.copy2(ROOT / source, DEST / target)
    packages = ["chronos-forecasting", "peft", "torch", "transformers", "huggingface-hub", "numpy"]
    (DEST / "requirements.txt").write_text("\n".join(f"{p}=={version(p)}" for p in packages) + "\n")
    for name, content in legal.items():
        (DEST / name).write_text(content)
    (DEST / "MODIFICATIONS.md").write_text(
        "Derived from amazon/chronos-2, revision " + REVISION + ".\n"
        "Changes: trained LoRA rank-8 updates for DE-LU electricity prices, merged into base weights.\n"
        "The original adapter is preserved separately; its base reference was made portable and pinned.\n"
        "This release adds inference utilities, interval calibration and model documentation.\n"
        "Original upstream license and any upstream NOTICE are retained.\n")
    manifest = dict(base_model="amazon/chronos-2", base_revision=REVISION,
                    legal_source=f"https://github.com/amazon-science/chronos-forecasting/tree/{source_revision}",
                    adapter_sha256=policy["adapter_sha256"],
                    calibration_sha256=policy["calibration_sha256"],
                    verification={"kind": "synthetic save/reload parity; no research outcomes accessed",
                                  "max_absolute_difference": float(np.max(np.abs(before - after))),
                                  "shape": list(after.shape)},
                    files={str(p.relative_to(DEST)): digest(p) for p in sorted(DEST.rglob("*")) if p.is_file()})
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified export: {DEST}")
    print(f"Total bytes: {sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file())}")


if __name__ == "__main__":
    main()
