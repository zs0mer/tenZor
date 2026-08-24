#!/usr/bin/env python3
import subprocess
import statistics
import os
import sys


def parse(text: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        if "] ns/op" not in line:  # skips bloat
            continue
        name = line.split("[")[0].strip()
        nums = [seg.split("]")[0] for seg in line.split("[")[1:]]
        out[name] = {
            "ns": float(nums[0]),
            "gflops": float(nums[1]) if len(nums) > 1 else 0,
        }
    return out


def run() -> dict[str, dict[str, float]]:
    out = subprocess.run(["./build/bench"], capture_output=True, text=True, check=True)
    print(".", end="", flush=True)
    return parse(out.stdout)


def scale_ns(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.1f} ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.1f} us"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.1f} ms"
    return f"{ns / 1_000_000_000:.2f} s"


def git_hash() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def format_results(data: dict[str, dict[str, float]], runs : int) -> str:
    lines: list[str] = [ f"Number of runs: {runs}" ]
    for name, measures in data.items():
        parts = [f"{name:37s}"]
        for measure, val in measures.items():
            if "GFlops" in measure:
                if val <= 0.001:
                    continue
                parts.append(f"{measure}: {val:10.1f} GF/s")
            else:
                parts.append(f"{measure}: {scale_ns(val):>12s}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def main(runs: int = 3):
    total = [run() for _ in range(runs)]

    result: dict[str, dict[str, float]] = {}
    for name in total[0]:
        values_ns = [data[name]["ns"] for data in total]
        values_gflops = [data[name]["gflops"] for data in total]

        result[name] = {
            "median": statistics.median(values_ns),
            "stdev": statistics.stdev(values_ns) if runs > 1 else 0.0,
            "GFlops median": statistics.median(values_gflops),
            "GFlops stdev": statistics.stdev(values_gflops) if runs > 1 else 0.0,
        }

    out = format_results(result, runs)
    print()          # newline after the progress dots
    print(out)

    os.makedirs("bench_results", exist_ok=True)
    with open(f"bench_results/{git_hash()}.txt", "w") as f:
        f.write(out)


if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    main(runs)
