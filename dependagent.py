"""
dependagent CLI
Usage:
    dependagent scan <path> [--no-ai] [--openai-model MODEL]

Example:
    export OPENAI_API_KEY="sk-..."
    python dependagent.py scan ./demo-project
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

import requests
import typer

load_dotenv()

# Optional OpenAI client (only used if --no-ai is False)
try:
    import openai
except Exception:
    openai = None  # we'll check later if user requested AI and openai is missing

app = typer.Typer(help="DependAgent — lightweight dependency scanner + AI migration guide MVP")
NPM_REGISTRY = "https://registry.npmjs.org"


def read_package_json(project_path: Path) -> Dict:
    pj = project_path / "package.json"
    if not pj.exists():
        typer.secho(f"package.json not found at {pj}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    with pj.open("r", encoding="utf-8") as f:
        return json.load(f)


def fetch_npm_info(pkg_name: str, retry: int = 2) -> Optional[Dict]:
    url = f"{NPM_REGISTRY}/{pkg_name}"
    for attempt in range(1, retry + 1):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
            else:
                typer.secho(f"Failed to fetch {pkg_name}: HTTP {r.status_code}", fg=typer.colors.YELLOW)
                return None
        except requests.RequestException as e:
            typer.secho(f"Network error fetching {pkg_name}: {e} (attempt {attempt})", fg=typer.colors.YELLOW)
            time.sleep(0.5)
    return None


def normalize_version(v: str) -> str:
    # crude normalization: remove leading ^ ~ >= <= etc.
    for prefix in ("^", "~", ">= ", "<= ", ">= ", "< ", "> "):
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
                "reason": "Failed to fetch npm info"
            })
            continue

        # get latest tag safely
        latest = None
        try:
            latest = info.get("dist-tags", {}).get("latest")
        except Exception:
            latest = None

        # try to find deprecation notice for the latest version (versions is a dict)
        deprecated = None
        try:
            versions = info.get("versions", {})
            if latest and isinstance(versions, dict) and latest in versions:
                deprecated = versions[latest].get("deprecated")
        except Exception:
            deprecated = None

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
            "npm_info": info  # included for optional deeper analysis later
        })
    return results


def build_ai_prompt(dep_name: str, current: str, latest: Optional[str], deprecated: Optional[str], npm_info: Dict) -> str:
    """
    Build a concise prompt sent to OpenAI to generate a migration guide in Markdown.
    Keep it explicit: current, target, any deprecation note, and a short request to provide:
      - upgrade steps (npm/yarn)
      - likely breaking changes
      - code example showing an idiomatic replacement (if applicable)
      - references (links) to primary sources (e.g., changelog or repo)
    """
    pkg_url = npm_info.get("repository", {}).get("url") if npm_info else None
    if isinstance(pkg_url, str) and pkg_url.startswith("git+"):
        pkg_url = pkg_url[len("git+") :]

    prompt = f"""You are an expert JavaScript/Node.js engineer helping to migrate dependencies.

Dependency: {dep_name}
Current version in project: {current}
Latest version (npm): {latest}
Deprecated notice (if any): {deprecated or 'None'}

Please produce a **practical Markdown migration guide** targeted at a maintainer who will:
- run one or two commands to upgrade,
- change a small snippet of code if needed,
- know if breaking changes exist (explicitly list them),
- provide an example code diff (use ```diff blocks),
- cite where to look in the upstream (changelog, migration guide, repo URL) using plain URLs when possible.

Use the npm registry metadata below as context (do not invent URLs). If you cannot confidently find migration steps, say so and recommend safe alternatives.

NPM metadata (JSON): {json.dumps({k: npm_info.get(k) for k in ['dist-tags','homepage','repository','bugs'] if npm_info}, ensure_ascii=False) if npm_info else '{}'}
"""
    return prompt


def call_openai_generate(prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 800) -> str:
    if openai is None:
        raise RuntimeError("openai package is not installed. Install dependencies or run with --no-ai.")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    openai.api_key = key

    # Use ChatCompletion (classic) interface — adjust if you want different endpoints
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        # older clients return choices[0].message.content; newer can differ
        text = resp["choices"][0]["message"]["content"]
        return text
    except Exception as e:
        raise RuntimeError(f"OpenAI request failed: {e}")


def generate_markdown_report(results: List[Dict], output_path: Path, use_ai: bool = True, model: str = "gpt-4o-mini"):
    md_lines = []
    md_lines.append("# Dependency Scan Report\n")
    md_lines.append(f"_Generated by dependagent - {time.strftime('%Y-%m-%d %H:%M:%S')}_\n")
    md_lines.append("---\n")

    for r in results:
        md_lines.append(f"## {r['name']}\n")
        md_lines.append(f"- Current: `{r['current']}`")
        md_lines.append(f"- Latest: `{r['latest']}`" if r.get("latest") else "- Latest: `unknown`")
        md_lines.append(f"- Status: **{r['status']}**\n")
        if r.get("deprecated"):
            md_lines.append(f"⚠️ **Deprecated**: {r.get('deprecated')}\n")

        # If outdated or deprecated and AI is enabled, ask AI for migration guide
        if use_ai and r["status"] in ("deprecated", "outdated"):
            try:
                typer.secho(f"Requesting AI migration guide for {r['name']} ...", fg=typer.colors.CYAN)
                prompt = build_ai_prompt(r["name"], r["current"], r.get("latest"), r.get("deprecated"), r.get("npm_info"))
                guide = call_openai_generate(prompt, model=model)
                md_lines.append("### Migration Guide (AI-generated)\n")
                md_lines.append(guide)
                md_lines.append("\n---\n")
            except Exception as e:
                typer.secho(f"AI generation failed for {r['name']}: {e}", fg=typer.colors.YELLOW)
                md_lines.append("### Migration Guide\n")
                md_lines.append(f"*AI generation failed: {e}*\n\n---\n")
        else:
            # Non-AI fallback: simple instructions
            if r["status"] == "outdated":
                md_lines.append("### Quick upgrade\n")
                md_lines.append(f"Run:\n```bash\nnpm install {r['name']}@{r.get('latest')}\n# or: yarn add {r['name']}@{r.get('latest')}\n```\n")
                md_lines.append("\n---\n")
            elif r["status"] == "deprecated":
                md_lines.append("### Recommendation\n")
                md_lines.append("This package is deprecated. Check the package repository or the npm page for suggested alternatives.\n\n---\n")

    report_text = "\n".join(md_lines)
    out_file = output_path / "DEPENDENCY_REPORT.md"
    out_file.write_text(report_text, encoding="utf-8")
    typer.secho(f"✅ Report written to {out_file}", fg=typer.colors.GREEN)


@app.command()
def scan(
    path: str = typer.Argument(..., help="Path to project folder containing package.json"),
    no_ai: bool = typer.Option(False, "--no-ai", help="Do not call OpenAI; produce minimal Markdown output"),
    openai_model: str = typer.Option("gpt-4o-mini", "--openai-model", help="OpenAI model to use"),
):
    """Scan a node project for outdated/deprecated deps and produce a Markdown migration guide."""
    project_path = Path(path).resolve()
    typer.secho(f"Scanning project: {project_path}", fg=typer.colors.BLUE)
    try:
        pkg = read_package_json(project_path)
    except Exception as e:
        typer.secho(f"Error reading package.json: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    deps = pkg.get("dependencies", {})
    if not deps:
        typer.secho("No dependencies found in package.json", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    typer.secho(f"Found {len(deps)} dependencies. Inspecting...", fg=typer.colors.BLUE)
    results = analyze_dependencies(deps)
    # Provide a short console summary
    for r in results:
        status_color = typer.colors.GREEN if r["status"] == "up-to-date" else typer.colors.RED
        typer.secho(f"- {r['name']}: {r['status']} (current: {r['current']}, latest: {r.get('latest')})", fg=status_color)

    # Generate markdown report (with or without AI)
    try:
        generate_markdown_report(results, project_path, use_ai=not no_ai, model=openai_model)
    except Exception as e:
        typer.secho(f"Failed to generate full report: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
