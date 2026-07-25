"""Read the single Task Duration(us) produced by ``msprof op``.

``msprof op`` performs warm-up internally and writes one selected operator to
``OPPROF_*/OpBasicInfo.csv``. This parser deliberately rejects zero or multiple
rows instead of silently mixing different operators or shapes.

Usage:
  python3 checker/get_time.py <msprof-output-root>
"""
import csv
import sys
from pathlib import Path

def read_task_duration(root: Path) -> float:
    csv_paths = sorted(root.rglob("OpBasicInfo.csv"))
    if len(csv_paths) != 1:
        raise RuntimeError(
            f"expected one OpBasicInfo.csv under {root}, found {len(csv_paths)}"
        )

    with csv_paths[0].open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    durations = [
        row.get("Task Duration(us)", "").strip()
        for row in rows
        if row.get("Task Duration(us)", "").strip()
    ]
    if len(durations) != 1:
        raise RuntimeError(
            f"expected one Task Duration(us) in {csv_paths[0]}, found {len(durations)}"
        )
    return float(durations[0])


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <msprof-output-root>", file=sys.stderr)
        sys.exit(2)
    try:
        duration = read_task_duration(Path(sys.argv[1]))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"{duration:.4f}")


if __name__ == "__main__":
    main()
