#!/usr/bin/env python3
"""Fail the build when a test file exists but the CI workflow never runs it.

A gate that silently stops covering a file reports the same green as one that covers
everything, so this compares the workflow's actual pytest invocation against the
test files on disk, and refuses to let a file go unwired without being named here.
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"

# Files deliberately NOT run in CI. Each needs a reason; the local pre-commit hook runs them.
LOCAL_ONLY = {
    "tests/test_app_smoke.py": "needs streamlit+langchain+chromadb; runs in the local pre-commit hook",
}


def pytest_invocation(text):
    """Return the argv of the `python -m pytest ...` step, or None."""
    for line in text.splitlines():
        m = re.search(r"python -m pytest\s+(.*)", line)
        if m:
            return m.group(1).split()
    return None


def classify(args, discovered):
    run, ignores, bad = set(), set(), []
    for token in args:
        if token.startswith("--ignore="):
            ignores.add(token.split("=", 1)[1].strip())
        elif token in ("-q", "-v", "--quiet", "--verbose"):
            continue
        elif token.startswith("-"):
            bad.append(token)
        else:
            p = token.rstrip("/")
            if p == "tests" or p == ".":
                run |= set(discovered)
            elif p in discovered:
                run.add(p)
            else:
                run.add(p)  # a path that does not exist: reported below
    return run - ignores, ignores, bad


def main():
    text = WORKFLOW.read_text()
    args = pytest_invocation(text)
    discovered = sorted(str(p.relative_to(REPO)) for p in (REPO / "tests").glob("test_*.py"))

    problems = []
    if args is None:
        problems.append("no `python -m pytest` invocation found in ci.yml")
        run, ignores, bad = set(), set(), []
    else:
        run, ignores, bad = classify(args, discovered)

    # 1. every discovered test file must either run, or be declared local-only with a reason
    unwired = [f for f in discovered if f not in run and f not in LOCAL_ONLY]
    for f in unwired:
        problems.append(f"test file not run by CI and not declared local-only: {f}")

    # 2. local-only declarations must be real files, and must actually be excluded
    for f, reason in LOCAL_ONLY.items():
        if not (REPO / f).exists():
            problems.append(f"local-only entry does not exist: {f}")
        if not reason.strip():
            problems.append(f"local-only entry has no reason: {f}")
        if f in run:
            problems.append(f"local-only entry is run by CI anyway: {f}")

    # 3. an --ignore of a file nobody declared is an undocumented exclusion
    for f in sorted(ignores - set(LOCAL_ONLY)):
        problems.append(f"ci.yml ignores {f} without declaring it local-only")

    # 4. arguments we do not understand could silently narrow the run
    for token in bad:
        problems.append(f"unrecognised pytest flag (may narrow the run): {token}")

    if problems:
        print("TEST WIRING FAILED")
        for p in problems:
            print("  -", p)
        print(f"\n  wired {len(run)}/{len(discovered)} discovered test files")
        return 1

    # 5. positive evidence, with a denominator: the suite must actually collect tests
    cmd = [sys.executable, "-m", "pytest", *args, "--collect-only", "-q"]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        print("TEST WIRING FAILED: collection errored, so a green run would prove nothing")
        print((proc.stdout or "")[-1200:], (proc.stderr or "")[-800:])
        return 1

    per_file = {}
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        m = re.match(r"^([\w./-]+\.py)::", line)          # default verbosity: path::test
        if m:
            per_file[m.group(1)] = per_file.get(m.group(1), 0) + 1
            continue
        m = re.match(r"^([\w./-]+\.py): (\d+)$", line)     # -q : path: N
        if m:
            per_file[m.group(1)] = per_file.get(m.group(1), 0) + int(m.group(2))
    collected = sum(per_file.values())

    if collected == 0:
        print("TEST WIRING FAILED: the suite collected zero tests")
        return 1

    empty = [f for f in sorted(run) if per_file.get(f, 0) == 0]
    if empty:
        print("TEST WIRING FAILED: wired but collecting zero tests")
        for f in empty:
            print("  -", f)
        return 1

    print(f"test wiring ok: {len(run)}/{len(discovered)} test files wired, {collected} tests collected")
    print("  per file: " + ", ".join(f"{f.split('/')[-1]}={per_file.get(f,0)}" for f in sorted(run)))
    for f, reason in LOCAL_ONLY.items():
        print(f"  local-only: {f} ({reason})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
