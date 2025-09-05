
#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pydriller import Repository

@dataclass
class RepoSpec:
    url: str
    name: str
    path: Path

def run(cmd: List[str], cwd: Optional[Path] = None, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check
    )

def ensure_cloned(url: str, workdir: Path) -> RepoSpec:
    name = url.rstrip('/').split('/')[-1].replace('.git', '')
    path = workdir / name
    if not path.exists():
        print(f"[clone] {url} -> {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        cp = run(["git", "clone", "--no-tags", "--depth", "1", url, str(path)])
        if cp.returncode != 0:
            print(cp.stderr, file=sys.stderr)
            raise RuntimeError(f"git clone failed for {url}")
        run(["git", "fetch", "--unshallow"], cwd=path)
    else:
        if (path / ".git").exists():
            run(["git", "remote", "set-url", "origin", url], cwd=path)
            run(["git", "fetch", "--all"], cwd=path)
        else:
            raise RuntimeError(f"Not a git repository: {path}")
    return RepoSpec(url=url, name=name, path=path)

def is_test_file(path: Optional[str]) -> bool:
    if not path:
        return False
    p = path.lower()
    import re
    return ("test" in p.split("/") or
            re.search(r"(^|/)(tests?|test_)\b", p) or
            re.search(r"(^|/)conftest\.py$", p))

SOURCE_EXTS = {
    ".c",".h",".cpp",".cc",".cxx",".hpp",
    ".java",".py",".rb",".go",".rs",".php",
    ".cs",".kt",".m",".mm",".swift",
    ".js",".jsx",".ts",".tsx",
    ".scala",".pl",".r",".jl",".lua",
    ".sh",".bash",".zsh",".ps1",
}

def is_source_code(path: Optional[str]) -> bool:
    if not path:
        return False
    return Path(path).suffix.lower() in SOURCE_EXTS

def is_readme(path: Optional[str]) -> bool:
    return path and Path(path).name.lower().startswith("readme")

def is_license(path: Optional[str]) -> bool:
    if not path:
        return False
    name = Path(path).name.lower()
    return name.startswith("license") or name.startswith("licence")

def classify_file(old_path: Optional[str], new_path: Optional[str]) -> str:
    p = (new_path or old_path or "").lower()
    if is_readme(p):
        return "readme"
    if is_license(p):
        return "license"
    if is_test_file(p):
        return "test"
    if is_source_code(p):
        return "source"
    return "other"

def git_diff_single(repo_dir: Path, parent_sha: str, commit_sha: str,
                    paths: List[str], algorithm: str) -> str:
    cmd = [
        "git","-c","core.safecrlf=false","diff",
        "--ignore-blank-lines","-w",
        f"--diff-algorithm={algorithm}",
        parent_sha, commit_sha,
        "--find-renames","--", *paths
    ]
    cp = run(cmd, cwd=repo_dir)
    return cp.stdout if cp.returncode == 0 else cp.stderr

def analyze_repo(spec: RepoSpec, max_commits: int, include_merges: bool) -> pd.DataFrame:
    rows = []
    seen = 0
    print(f"[analyze] {spec.name} at {spec.path}")
    for commit in Repository(str(spec.path), order="reverse",
                             only_no_merge=not include_merges).traverse_commits():
        if max_commits and seen >= max_commits:
            break
        seen += 1
        parent_sha = commit.parents[0] if commit.parents else "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
        for m in commit.modified_files:
            paths = list({p for p in [m.old_path, m.new_path] if p})
            if not paths:
                continue
            diff_myers = git_diff_single(spec.path, parent_sha, commit.hash, paths, "myers")
            diff_hist  = git_diff_single(spec.path, parent_sha, commit.hash, paths, "histogram")
            discrep = "No" if diff_myers == diff_hist else "Yes"
            ftype = classify_file(m.old_path, m.new_path)
            rows.append({
                "repo": spec.name,
                "old_file_path": m.old_path or "",
                "new_file_path": m.new_path or "",
                "commit_sha": commit.hash,
                "parent_commit_sha": parent_sha,
                "commit_message": commit.msg.strip().replace("\n"," "),
                "file_type": ftype,
                "diff_myers": diff_myers,
                "diff_hist": diff_hist,
                "Discrepancy": discrep,
            })
    return pd.DataFrame(rows)

def build_and_save(repo_urls: List[str], workdir: Path,
                   max_commits: int, out_csv: Path,
                   include_merges: bool, plots_dir: Optional[Path] = None):
    workdir.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for url in repo_urls:
        spec = ensure_cloned(url, workdir)
        df = analyze_repo(spec, max_commits, include_merges)
        all_dfs.append(df)
    full = pd.concat(all_dfs, ignore_index=True)
    full.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
    print(f"[ok] Dataset written: {out_csv} ({len(full)} rows)")
    stats = (full.groupby(["file_type","Discrepancy"]).size()
             .reset_index(name="count")
             .pivot(index="file_type", columns="Discrepancy", values="count")
             .fillna(0).astype(int))
    stats.to_csv(out_csv.with_name(out_csv.stem + "_stats.csv"))
    if plots_dir:
        plots_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
        mismatches = full[full["Discrepancy"]=="Yes"].groupby("file_type").size()
        mismatches.plot(kind="bar", title="#Mismatches by file type")
        plt.tight_layout()
        plt.savefig(plots_dir/"mismatches_by_type.png")
        for cat in ["source","test","readme","license"]:
            cnt = (full[(full["file_type"]==cat)&(full["Discrepancy"]=="Yes")]).shape[0]
            plt.figure()
            plt.bar([cat],[cnt])
            plt.title(f"#Mismatches for {cat}")
            plt.tight_layout()
            plt.savefig(plots_dir/f"mismatches_{cat}.png")

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--repos", nargs="+", required=True)
    p.add_argument("--workdir", type=Path, default=Path("repos"))
    p.add_argument("--max-commits", type=int, default=300)
    p.add_argument("--out", type=Path, default=Path("diff_dataset.csv"))
    p.add_argument("--include-merges", action="store_true")
    p.add_argument("--plots-dir", type=Path, default=None)
    return p.parse_args(argv)

def main():
    args = parse_args()
    build_and_save(args.repos, args.workdir, args.max_commits,
                   args.out, args.include_merges, args.plots_dir)

if __name__ == "__main__":
    main()
