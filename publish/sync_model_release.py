"""Update documentation/source bundle of an existing model; original weight hashes must match."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from huggingface_hub import HfApi, ModelCard

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "exports/chronos-2-de-lu-pregate"
SOURCE = ROOT / "exports/github/chronos-2-de-lu-pregate"
REPO_ID = "AryanSudhirDev/chronos-2-de-lu-pregate"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    previous = json.loads((MODEL / "manifest.json").read_text())
    for name in ["model.safetensors", "config.json", "adapter/adapter_model.safetensors", "calibration.json"]:
        if digest(MODEL / name) != previous["files"][name]:
            raise RuntimeError(f"Frozen release artifact changed: {name}")
    for source, destination in [("publish/model_card.md", "README.md"),
                                ("publish/inference.py", "inference.py"),
                                ("scripts/make_example_input.py", "make_example_input.py"),
                                ("docs/results.png", "results.png"), ("CITATION.cff", "CITATION.cff")]:
        shutil.copy2(ROOT / source, MODEL / destination)
    for p in (ROOT / "docs").glob("*"):
        if p.is_file():
            (MODEL / "docs").mkdir(exist_ok=True)
            shutil.copy2(p, MODEL / "docs" / p.name)
    shutil.copy2(SOURCE / "NOTICE", MODEL / "NOTICE")
    # Archive committed files only: no .git, environments, private logs or local caches.
    subprocess.run(["git", "archive", "--format=zip", "--output", str(MODEL / "source-release.zip"), "HEAD"], cwd=SOURCE, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    previous["source_repository"] = "https://github.com/" + REPO_ID
    previous["source_commit"] = commit
    previous["files"] = {str(p.relative_to(MODEL)): digest(p) for p in sorted(MODEL.rglob("*"))
                         if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts}
    (MODEL / "manifest.json").write_text(json.dumps(previous, indent=2) + "\n")
    ModelCard.load(MODEL / "README.md").validate()
    api = HfApi(token=args.token_file.read_text().strip())
    if api.whoami()["name"].lower() != "aryansudhirdev":
        raise RuntimeError("Credential belongs to another account")
    files = sorted([*previous["files"], "manifest.json"])
    update = api.upload_folder(repo_id=REPO_ID, folder_path=MODEL, allow_patterns=files,
                               commit_message="Add comprehensive research documentation, source release and complete example")
    info = api.model_info(REPO_ID, revision=update.oid, files_metadata=True)
    remote = {p.rfilename:p for p in info.siblings}
    for name in files:
        if name not in remote or remote[name].size != (MODEL / name).stat().st_size:
            raise RuntimeError(f"Remote file missing or size mismatch: {name}")
        if remote[name].lfs and remote[name].lfs.sha256 != digest(MODEL / name):
            raise RuntimeError(f"Remote hash mismatch: {name}")
    result = dict(url="https://huggingface.co/" + REPO_ID, model_commit=update.oid,
                  source_commit=commit, verified_files=len(files), frozen_weights_unchanged=True)
    (ROOT / "exports/huggingface-release-update.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
