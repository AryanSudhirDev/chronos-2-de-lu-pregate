"""Publish the verified export using a secure token file, never a command-line token."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / "exports/chronos-2-de-lu-pregate"
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Release hash mismatch: {name}")
    api = HfApi(token=args.token_file.read_text().strip())
    owner = "AryanSudhirDev"
    identity = api.whoami()
    if identity["name"].lower() != owner.lower():
        raise RuntimeError(f"Authenticated account is not the requested owner {owner}")
    repo_id = f"{owner}/chronos-2-de-lu-pregate"
    # Refuse to overwrite any pre-existing repository.
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=False)
    print(f"Created https://huggingface.co/{repo_id}", flush=True)
    files = sorted([*manifest["files"], "manifest.json"])
    commit = api.upload_folder(repo_id=repo_id, folder_path=folder, allow_patterns=files,
                               commit_message="Publish frozen pre-gate Chronos-2 model, adapter and inference")
    info = api.model_info(repo_id, revision=commit.oid, files_metadata=True)
    remote = {entry.rfilename: entry for entry in info.siblings}
    for name in files:
        if name not in remote or remote[name].size != (folder / name).stat().st_size:
            raise RuntimeError(f"Remote file missing or size mismatch: {name}")
        if remote[name].lfs:
            if remote[name].lfs.sha256 != hashlib.sha256((folder / name).read_bytes()).hexdigest():
                raise RuntimeError(f"Remote LFS hash mismatch: {name}")
    result = dict(repo_id=repo_id, url=f"https://huggingface.co/{repo_id}",
                  commit=commit.oid, verified_files=files, private=info.private)
    (root / "exports/huggingface-publication.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
