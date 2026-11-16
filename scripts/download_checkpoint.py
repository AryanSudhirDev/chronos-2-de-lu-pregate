"""Install the published frozen adapter at the research pipeline's expected location.

Downloads public weights only. Does not train, collect feeds or issue forecasts.
"""
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "AryanSudhirDev/chronos-2-de-lu-pregate"
MODEL_REVISION = "70aec15186fd38d649cb6145a2f44dba68c9fa75"
BASE_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


def main():
    policy = json.loads((ROOT / "runs/paper/policy.json").read_text())
    source = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                    allow_patterns=["adapter/*", "calibration.json"]))
    base = snapshot_download("amazon/chronos-2", revision=BASE_REVISION,
                             allow_patterns=["config.json", "model.safetensors"])
    weight = source / "adapter/adapter_model.safetensors"
    if hashlib.sha256(weight.read_bytes()).hexdigest() != policy["adapter_sha256"]:
        raise RuntimeError("Published adapter does not match the frozen policy")
    if hashlib.sha256((source / "calibration.json").read_bytes()).hexdigest() != policy["calibration_sha256"]:
        raise RuntimeError("Published calibration does not match the frozen policy")
    dest = ROOT / policy["adapter"]
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / weight.name
    if target.exists():
        if target.read_bytes() != weight.read_bytes():
            raise FileExistsError("Existing adapter differs; refusing to overwrite")
    else:
        shutil.copy2(weight, target)
    config = json.loads((source / "adapter/adapter_config.json").read_text())
    config["base_model_name_or_path"] = base
    config["revision"] = None  # exact local snapshot is already pinned
    target_config = dest / "adapter_config.json"
    if not target_config.exists():
        target_config.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Frozen adapter available at {dest}")
    print("No research, feed collection or paper processes started.")


if __name__ == "__main__":
    main()
