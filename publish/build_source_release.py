"""Build an allowlisted public source snapshot. Does not publish or run research."""
from importlib.metadata import distributions, version
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "exports/github/chronos-2-de-lu-pregate"


def copy(source, target=None):
    destination = DEST / (target or source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / source, destination)


def main():
    DEST.mkdir(parents=True, exist_ok=False)
    for directory in ("src/epf", "tests", "scripts", "publish", "docs"):
        for p in sorted((ROOT / directory).rglob("*")):
            if p.is_file() and p.suffix in {".py", ".sh", ".md", ".png"} and "__pycache__" not in p.parts:
                if "crypto" not in p.relative_to(ROOT).parts:
                    copy(str(p.relative_to(ROOT)))
    for name in ("pyproject.toml", "CITATION.cff", "CONTRIBUTING.md", ".github/workflows/tests.yml"):
        copy(name)
    copy("publish/github_readme.md", "README.md")
    notes = (ROOT / "README.md").read_text()
    notes = notes.replace("[interactive report](runs/replay/report.html)", "interactive report (local historical HTML, not redistributed)")
    notes = notes.replace("[hourly decisions](runs/replay/hourly.csv)", "hourly decisions (local historical CSV, not redistributed)")
    (DEST / "RESEARCH_NOTES.md").write_text(
        "> Historical study notes. Scheduling/future-tense statements below are preserved from\n"
        "> development and are superseded by README.md and docs/RESULTS.md. The paper stream stopped.\n\n" + notes)
    # Retain original attribution and Apache license already fetched for the model.
    for name in ("LICENSE", "NOTICE"):
        copy(f"exports/chronos-2-de-lu-pregate/{name}", name)
    with (DEST / "NOTICE").open("a") as f:
        f.write("\nDE-LU forecasting research and release additions\nCopyright 2026 Aryan Sudhir\n")
    (DEST / ".gitignore").write_text(
        ".venv/\n__pycache__/\n*.pyc\n*.egg-info/\n.DS_Store\ndata/\nmodel/\nexports/\n"
        "example_input.json\nforecast.json\n.env\n.env.*\n*.pem\n*.key\nruns/ckpt/\nruns/paper/*.lock\n")
    selected = ["runs/ledger.jsonl", "runs/compare.md", "runs/compare_holdout.md",
                "runs/holdout_summary.json", "runs/holdout_audit_manifest.json", "runs/holdout_issued_audit.zip",
                "runs/readiness/protocol.json", "runs/readiness/calibration.json",
                "runs/readiness/report.md", "runs/readiness/summary.json",
                "runs/adaptive/protocol.json", "runs/adaptive/report.md", "runs/adaptive/summary.json",
                "runs/replay/protocol.json", "runs/replay/report.md", "runs/replay/diagnosis.md",
                "runs/replay/summary.json", "runs/paper/policy.json", "runs/paper/summary.json"]
    for name in selected:
        copy(name)
    for p in sorted((ROOT / "runs/forecasts").glob("*.npz")):
        copy(str(p.relative_to(ROOT)))
    copy("runs/STOP")
    direct = ["pandas", "pyarrow", "numpy", "scipy", "scikit-learn", "lightgbm", "requests",
              "holidays", "torch", "chronos-forecasting", "peft", "matplotlib", "huggingface-hub"]
    (DEST / "requirements-research.txt").write_text("\n".join(f"{n}=={version(n)}" for n in direct) + "\n")
    copy("exports/chronos-2-de-lu-pregate/requirements.txt", "requirements-inference.txt")
    observed = sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions() if d.metadata['Name'] != "epf"})
    (DEST / "environment-observed.txt").write_text("# Observed macOS Python 3.12 environment; not a portable lockfile\n" + "\n".join(observed) + "\n")
    manifest = {str(p.relative_to(DEST)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(DEST.rglob("*")) if p.is_file()}
    (DEST / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {len(manifest)} source/research files at {DEST}")


if __name__ == "__main__":
    main()
