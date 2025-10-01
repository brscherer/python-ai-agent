#!/usr/bin/env python3
"""
Simple DependAgent CLI (argparse-based)

Usage:
    python dependagent.py scan <path> [--no-ai] [--openai-model MODEL]

Examples:
    python dependagent.py scan ./demo-project
    python dependagent.py scan ./demo-project --no-ai
    python dependagent.py scan ./demo-project --openai-model gpt-4o-mini
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

import requests

load_dotenv()

# Optional OpenAI client (only used if --no-ai is False)
try:
    import openai
except Exception:
    openai = None

NPM_REGISTRY = "https://registry.npmjs.org"


def read_package_json(project_path: Path) -> Dict:
    pj = project_path / "package.json"
    if not pj.exists():
        print(f"[ERROR] package.json not found at {pj}", file=sys.stderr)
        sys.exit(1)
    with pj.open("r", encoding="utf-8") as f:
        return json.load(f)


def fetch_npm_info(pkg_name: str, timeout: float = 10.0) -> Optional[Dict]:
    url = f"{NPM_REGISTRY}/{pkg_name}"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"[WARN] npm registry returned HTTP {r.status_code} for {pkg_name}", file=sys.stderr)
            return None
    except requests.RequestException as e:
        print(f"[WARN] network error fetching {pkg_name}: {e}", file=sys.stderr)
        return None


def normalize_version(v: str) -> str:
    # simple normalization removing ^ ~ >= <= etc.
    if not isinstance(v, str):
        return str(v)
    for prefix in ("^", "~", ">=", "<=", ">", "<", "="):
        if v.startswith(prefix):
            v = v[len(prefix):]
    return v.strip()


def analyze_dependencies(deps: Dict[str, str]) -> List[Dict]:
    results = []
    for dep, cur_version in deps.items():
        info = fetch_npm_info(dep)
        if info is None:
            results.append({
                "name": dep,
                "current": cur_version,
                "latest": None,
                "status": "unknown",
                "deprecated": None,
                "npm_info": None
            })
            continue

        latest = info.get("dist-tags", {}).get("latest")
        versions = info.get("versions", {})
        deprecated = None
        # Check for deprecation message on the latest version if present.
        if latest and isinstance(versions, dict) and latest in versions:
            deprecated = versions[latest].get("deprecated")

        current_norm = normalize_version(cur_version)
        status = "up-to-date"
        if deprecated:
            status = "deprecated"
        elif latest and current_norm != latest:
            status = "outdated"

        results.append({
            "name": dep,
            "current": cur_version,
            "latest": latest,
            "status": status,
            "deprecated": deprecated,
            "npm_info": info
        })
    return results


def build_ai_prompt(dep_name: str, current: str, latest: Optional[str], deprecated: Optional[str], npm_info: Optional[Dict]) -> str:
    pkg_meta = {}
    if npm_info:
        for k in ("dist-tags", "homepage", "repository", "bugs"):
            if k in npm_info:
                pkg_meta[k] = npm_info[k]
    prompt = f"""
You are an expert JavaScript/Node.js engineer helping to migrate dependencies.

Dependency: {dep_name}
Current version in project: {current}
Latest version (npm): {latest or 'unknown'}
Deprecated notice (if any): {deprecated or 'None'}

Please produce a practical Markdown migration guide targeted at a maintainer:
- include the CLI command(s) to upgrade (npm/yarn),
- list any likely breaking changes and how to adapt code,
- provide a small example code diff (use ```diff),
- include references to changelog/migration guide/repo if available (plain URLs).

NPM metadata: {json.dumps(pkg_meta, ensure_ascii=False)}
"""
    return prompt


def call_openai_generate(prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 700) -> str:
    if openai is None:
        raise RuntimeError("OpenAI library is not installed (install 'openai' in requirements), or it's not importable.")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    openai.api_key = key

    # Use ChatCompletion classic interface for broad compatibility
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    # extract text
    if "choices" in resp and len(resp["choices"]) > 0:
        # older and newer clients vary shape; handle common case
        choice = resp["choices"][0]
        if isinstance(choice.get("message"), dict):
            return choice["message"].get("content", "").strip()
        return choice.get("text", "").strip()
    raise RuntimeError("OpenAI response did not contain choices.")


def generate_markdown_report(results: List[Dict], output_path: Path, use_ai: bool = True, openai_model: str = "gpt-4o-mini") -> None:
    lines: List[str] = []
    lines.append("# Dependency Scan Report")
    lines.append(f"_Generated by dependagent at {time.strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("---\n")

    for r in results:
        lines.append(f"## {r['name']}")
        lines.append(f"- Current: `{r['current']}`")
        lines.append(f"- Latest: `{r.get('latest')}`" if r.get('latest') else "- Latest: `unknown`")
        lines.append(f"- Status: **{r['status']}**")
        if r.get("deprecated"):
            lines.append(f"⚠️ **Deprecated**: {r.get('deprecated')}")
        lines.append("")

        if use_ai and r["status"] in ("deprecated", "outdated"):
            try:
                print(f"[INFO] requesting AI migration guide for {r['name']} ...")
                prompt = build_ai_prompt(r["name"], r["current"], r.get("latest"), r.get("deprecated"), r.get("npm_info"))
                guide = call_openai_generate(prompt, model=openai_model)
                lines.append("### Migration Guide (AI-generated)\n")
                lines.append(guide)
            except Exception as e:
                print(f"[WARN] AI generation failed for {r['name']}: {e}", file=sys.stderr)
                lines.append("### Migration Guide\n")
                lines.append(f"*AI generation failed: {e}*")
        else:
            # fallback simple guidance
            if r["status"] == "outdated" and r.get("latest"):
                lines.append("### Quick upgrade")
                lines.append("Run:")
                lines.append("```bash")
                lines.append(f"npm install {r['name']}@{r.get('latest')}")
                lines.append("# or: yarn add {r['name']}@{r.get('latest')}")
                lines.append("```")
            elif r["status"] == "deprecated":
                lines.append("### Recommendation")
                lines.append("This package is deprecated. Check its npm page or repository for recommended alternatives.")
        lines.append("\n---\n")

    out_file = output_path / "DEPENDENCY_REPORT.md"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] report written to {out_file}")


def cmd_scan(path: str, no_ai: bool, openai_model: str) -> None:
    project_path = Path(path).resolve()
    print(f"[INFO] scanning project: {project_path}")
    pkg = read_package_json(project_path)
    deps = pkg.get("dependencies", {})
    if not deps:
        print("[INFO] no dependencies found in package.json")
        return

    print(f"[INFO] found {len(deps)} dependencies")
    results = analyze_dependencies(deps)

    # print quick summary
    for r in results:
        status = r["status"]
        print(f"- {r['name']}: {status} (current {r['current']}, latest {r.get('latest')})")

    generate_markdown_report(results, project_path, use_ai=not no_ai, openai_model=openai_model)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="dependagent", description="DependAgent — scan node project deps and produce a migration report")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="scan a project folder")
    scan_parser.add_argument("path", help="path to project folder containing package.json")
    scan_parser.add_argument("--no-ai", action="store_true", help="do not call OpenAI; produce minimal markdown")
    scan_parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model to use if AI enabled")

    args = parser.parse_args(argv)

    if args.command == "scan":
        cmd_scan(args.path, args.no_ai, args.openai_model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
