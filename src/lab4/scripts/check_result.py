#!/usr/bin/env python3
import argparse
import ast
import math
import os
import sys
from pathlib import Path

BH_COLUMNS = 7
CONSTRAINT_COLUMNS = 8
TIME_TOLERANCE = 1.0e-8
DENOMINATOR_EPSILON = 1.0e-12
RMS_LIMIT = 1.0e-3
CONSTRAINT_LIMIT = 2.0


class CheckError(Exception):
    pass


def parse_nonnegative_finite(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return number


def exceeds_limit(value, limit):
    return value > limit and not math.isclose(
        value, limit, rel_tol=1.0e-12, abs_tol=0.0
    )


def read_rows(path, expected_columns):
    rows = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                line = raw_line.split("#", 1)[0].strip()
                if not line:
                    continue
                fields = line.split()
                if len(fields) != expected_columns:
                    raise CheckError(
                        f"{path}:{line_number}: expected {expected_columns} columns, "
                        f"found {len(fields)}"
                    )
                try:
                    row = [float(field) for field in fields]
                except ValueError as exc:
                    raise CheckError(
                        f"{path}:{line_number}: invalid numeric value"
                    ) from exc
                if not all(math.isfinite(value) for value in row):
                    raise CheckError(
                        f"{path}:{line_number}: non-finite value"
                    )
                rows.append(row)
    except OSError as exc:
        raise CheckError(f"cannot read {path}: {exc}") from exc
    except UnicodeError as exc:
        raise CheckError(f"cannot decode {path} as UTF-8: {exc}") from exc

    if not rows:
        raise CheckError(f"{path}: no data rows")
    return rows


def read_trajectory(path):
    rows = read_rows(path, BH_COLUMNS)
    previous_time = None
    for row_number, row in enumerate(rows, 1):
        time = row[0]
        if previous_time is not None and time <= previous_time:
            raise CheckError(
                f"{path}: data row {row_number}: times must be strictly "
                "increasing with no duplicates"
            )
        previous_time = time
    return rows


def match_trajectory_times(reference, target, tolerance):
    matches = []
    target_index = 0

    for reference_row in reference:
        reference_time = reference_row[0]
        while (
            target_index < len(target)
            and target[target_index][0] < reference_time
            and reference_time - target[target_index][0] > tolerance
        ):
            target_index += 1

        if (
            target_index >= len(target)
            or abs(target[target_index][0] - reference_time) > tolerance
        ):
            raise CheckError(
                f"time coverage incomplete: matched {len(matches)}/{len(reference)}; "
                f"no unused target time within {tolerance:.3g} of "
                f"reference time {reference_time:.17g}"
            )

        matches.append((reference_row, target[target_index]))
        target_index += 1

    return matches


def calculate_trajectory_rms(matches):
    squared_errors = []
    for reference_row, target_row in matches:
        for reference_value, target_value in zip(reference_row[1:], target_row[1:]):
            denominator = max(abs(reference_value), abs(target_value))
            if denominator <= DENOMINATOR_EPSILON:
                continue
            relative_error = (target_value - reference_value) / denominator
            squared_errors.append(relative_error * relative_error)

    if not squared_errors:
        raise CheckError(
            f"no effective coordinate terms above denominator epsilon "
            f"{DENOMINATOR_EPSILON:.3g}"
        )

    rms = math.sqrt(math.fsum(squared_errors) / len(squared_errors))
    return rms, len(squared_errors)


def read_level_zero_constraints(path):
    rows = read_rows(path, CONSTRAINT_COLUMNS)
    level_zero_rows = []
    group_sizes = []
    current_time = None
    current_size = 0

    for row_number, row in enumerate(rows, 1):
        time = row[0]
        if current_time is None or time > current_time:
            if current_time is not None:
                group_sizes.append(current_size)
            # Levels are written consecutively from level 0, so the first row is level 0.
            level_zero_rows.append(row)
            current_time = time
            current_size = 1
        elif time == current_time:
            current_size += 1
        else:
            raise CheckError(
                f"{path}: data row {row_number}: time groups must be strictly "
                "increasing and contiguous"
            )

    group_sizes.append(current_size)
    expected_levels = group_sizes[0]
    for group_number, size in enumerate(group_sizes, 1):
        if size != expected_levels:
            raise CheckError(
                f"{path}: irregular level count at time group {group_number}: "
                f"expected {expected_levels}, found {size}"
            )

    return level_zero_rows, expected_levels


def constraint_maxima(level_zero_rows):
    return [
        max(abs(row[column]) for row in level_zero_rows)
        for column in range(1, 5)
    ]


def read_config_string(config_path, variable_name):
    try:
        source = config_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(config_path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise CheckError(f"cannot parse {config_path}: {exc}") from exc

    value = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in node.targets
        ):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError) as exc:
                raise CheckError(
                    f"{config_path}: {variable_name} must be a string literal"
                ) from exc

    if not isinstance(value, str) or not value:
        raise CheckError(f"{config_path}: missing non-empty {variable_name}")
    return value


def resolve_path(root, value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def unique_candidates(paths):
    result = []
    for path in paths:
        if path not in result:
            result.append(path)
    return result


def locate_result_directory(path):
    candidates = unique_candidates(
        [
            path,
            path / "AMSS_NCKU_output",
            path / "binary_output",
            path / "AMSS_NCKU_output" / "binary_output",
        ]
    )
    for candidate in candidates:
        if (
            (candidate / "bssn_BH.dat").is_file()
            and (candidate / "bssn_constraint.dat").is_file()
        ):
            return candidate
    raise CheckError(
        f"cannot find bssn_BH.dat and bssn_constraint.dat under {path}"
    )


def locate_golden_directory(path):
    candidates = unique_candidates(
        [
            path,
            path / "AMSS_NCKU_output",
            path / "binary_output",
            path / "AMSS_NCKU_output" / "binary_output",
        ]
    )
    for candidate in candidates:
        if (candidate / "bssn_BH.dat").is_file():
            return candidate
    raise CheckError(f"cannot find bssn_BH.dat under golden directory {path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Check Lab4 black-hole trajectories and level-0 constraints."
    )
    parser.add_argument("result_dir", nargs="?")
    parser.add_argument("golden_dir", nargs="?")
    parser.add_argument(
        "--time-tolerance",
        type=parse_nonnegative_finite,
        default=TIME_TOLERANCE,
        help=f"absolute trajectory time tolerance (default: {TIME_TOLERANCE:g})",
    )
    return parser


def main():
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parent.parent

    # AMSS_OUTPUT_ROOT (default: lab root) is the parent of the run
    # directory. Default RESULT_DIR and relative explicit RESULT_DIR both
    # resolve against it, mirroring run.sh semantics. Absolute paths are
    # kept as-is.
    output_root_env = os.environ.get("AMSS_OUTPUT_ROOT")
    if output_root_env:
        output_root = Path(output_root_env).expanduser().resolve()
    else:
        output_root = root

    try:
        if args.result_dir is None:
            file_directory = read_config_string(
                root / "AMSS_NCKU_Input.py", "File_directory"
            )
            # Absolute File_directory is honored; relative File_directory
            # resolves against AMSS_OUTPUT_ROOT (or lab root if unset).
            result_path = resolve_path(output_root, file_directory) / "AMSS_NCKU_output"
        else:
            # Explicit RESULT_DIR: absolute paths kept; relative paths
            # resolve against AMSS_OUTPUT_ROOT (or lab root if unset).
            result_path = resolve_path(output_root, args.result_dir)

        # GOLDEN_DIR stays relative to the lab root so the shipped golden
        # data is found regardless of where the run was written.
        golden_path = resolve_path(root, args.golden_dir or "golden")
        result_directory = locate_result_directory(result_path)
        golden_directory = locate_golden_directory(golden_path)
    except CheckError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("FINAL: FAIL")
        return 1

    reference_path = golden_directory / "bssn_BH.dat"
    target_path = result_directory / "bssn_BH.dat"
    constraint_path = result_directory / "bssn_constraint.dat"
    failed = False

    print(f"Golden: {golden_directory}")
    print(f"Result: {result_directory}")

    try:
        reference = read_trajectory(reference_path)
        target = read_trajectory(target_path)
        matches = match_trajectory_times(reference, target, args.time_tolerance)
        rms, effective_terms = calculate_trajectory_rms(matches)
        print(
            f"Trajectory: matched times {len(matches)}/{len(reference)}, "
            f"effective terms {effective_terms}"
        )
        print(f"Trajectory RMS: {rms:.9g} ({rms * 100.0:.6f}%)")
        if exceeds_limit(rms, RMS_LIMIT):
            print(
                f"Trajectory: FAIL - RMS exceeds {RMS_LIMIT:.6g} "
                f"({RMS_LIMIT * 100.0:.6f}%)"
            )
            failed = True
        else:
            print(
                f"Trajectory: PASS - RMS <= {RMS_LIMIT:.6g} "
                f"({RMS_LIMIT * 100.0:.6f}%)"
            )
    except CheckError as exc:
        print(f"Trajectory: FAIL - {exc}")
        failed = True

    try:
        level_zero_rows, level_count = read_level_zero_constraints(constraint_path)
        maxima = constraint_maxima(level_zero_rows)
        print(
            f"Constraints: {len(level_zero_rows)} time groups, "
            f"{level_count} levels per group"
        )
        print(
            "Constraint maxima (level 0): "
            + ", ".join(
                f"{name}={value:.9g}"
                for name, value in zip(("Ham", "Px", "Py", "Pz"), maxima)
            )
        )
        exceeded = [
            name
            for name, value in zip(("Ham", "Px", "Py", "Pz"), maxima)
            if exceeds_limit(value, CONSTRAINT_LIMIT)
        ]
        if exceeded:
            print(
                f"Constraints: FAIL - {', '.join(exceeded)} exceed "
                f"{CONSTRAINT_LIMIT:.6g}"
            )
            failed = True
        else:
            print(f"Constraints: PASS - all maxima <= {CONSTRAINT_LIMIT:.6g}")
    except CheckError as exc:
        print(f"Constraints: FAIL - {exc}")
        failed = True

    print(f"FINAL: {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
