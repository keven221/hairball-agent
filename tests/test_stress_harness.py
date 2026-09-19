from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from tests.stress_harness import (
    ProcessSpec,
    SubprocessSpec,
    ThreadSpec,
    checked_thread_target,
    collect_worker_results,
    run_worker_process,
    wait_for_processes,
    wait_for_subprocesses,
    wait_for_threads,
)


def _ok_worker(value: int) -> list[dict[str, int]]:
    return [{"value": value}]


def _failing_worker() -> list[dict[str, object]]:
    raise RuntimeError("worker exploded")


def _sleeping_worker() -> list[dict[str, object]]:
    time.sleep(30)
    return []


def test_process_success_envelope_round_trip(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    result_file = str(tmp_path / "ok.json")
    process = ctx.Process(
        target=run_worker_process,
        args=("worker-0", result_file, _ok_worker, (7,)),
    )
    spec = ProcessSpec("worker-0", process, result_file)
    process.start()

    assert wait_for_processes([spec], timeout_s=10) == []
    events, failures = collect_worker_results([spec])
    assert failures == []
    assert events == [{"value": 7}]


def test_process_exception_is_not_silenced(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    result_file = str(tmp_path / "failure.json")
    process = ctx.Process(
        target=run_worker_process,
        args=("worker-fail", result_file, _failing_worker, ()),
    )
    spec = ProcessSpec("worker-fail", process, result_file)
    process.start()

    lifecycle = wait_for_processes([spec], timeout_s=10)
    _, envelope = collect_worker_results([spec])

    assert any("PROCESS EXIT" in failure for failure in lifecycle)
    assert any(
        "WORKER ERROR: worker-fail: RuntimeError: worker exploded" in failure
        for failure in envelope
    )


def test_missing_and_corrupt_results_are_failures(tmp_path: Path) -> None:
    missing = ProcessSpec("missing", object(), str(tmp_path / "missing.json"))
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("{", encoding="utf-8")
    corrupt = ProcessSpec("corrupt", object(), str(corrupt_path))

    events, failures = collect_worker_results([missing, corrupt])

    assert events == []
    assert any("MISSING RESULT: missing" in failure for failure in failures)
    assert any("INVALID RESULT: corrupt" in failure for failure in failures)


def test_timeout_terminates_process_and_fails(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    result_file = str(tmp_path / "timeout.json")
    process = ctx.Process(
        target=run_worker_process,
        args=("worker-timeout", result_file, _sleeping_worker, ()),
    )
    spec = ProcessSpec("worker-timeout", process, result_file)
    process.start()

    failures = wait_for_processes([spec], timeout_s=0.1)

    assert any("PROCESS TIMEOUT: worker-timeout" in failure for failure in failures)
    assert not process.is_alive()


def test_thread_exception_and_timeout_are_failures() -> None:
    errors: "queue.Queue[dict[str, str]]" = queue.Queue()

    def explode() -> None:
        raise ValueError("thread exploded")

    def wait_forever() -> None:
        time.sleep(30)

    failing = threading.Thread(
        target=checked_thread_target,
        args=("thread-fail", errors, explode),
        daemon=True,
    )
    sleeping = threading.Thread(
        target=checked_thread_target,
        args=("thread-timeout", errors, wait_forever),
        daemon=True,
    )
    failing.start()
    sleeping.start()

    failures = wait_for_threads(
        [
            ThreadSpec("thread-fail", failing),
            ThreadSpec("thread-timeout", sleeping),
        ],
        timeout_s=0.1,
        errors=errors,
    )

    assert any(
        "THREAD ERROR: thread-fail: ValueError: thread exploded" in failure
        for failure in failures
    )
    assert any(
        "THREAD TIMEOUT: thread-timeout" in failure for failure in failures
    )


def test_error_envelope_is_structured_json(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    result_file = str(tmp_path / "failure.json")
    process = ctx.Process(
        target=run_worker_process,
        args=("worker-fail", result_file, _failing_worker, ()),
    )
    process.start()
    process.join(timeout=10)

    payload = json.loads(Path(result_file).read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["error_type"] == "RuntimeError"
    assert "worker exploded" in payload["traceback"]


def test_real_subprocess_exit_and_timeout_are_failures() -> None:
    failed = subprocess.Popen(
        [sys.executable, "-c", "raise SystemExit(7)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    sleeping = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    failures = wait_for_subprocesses(
        [
            SubprocessSpec("failed", failed),
            SubprocessSpec("sleeping", sleeping),
        ],
        timeout_s=0.2,
    )

    assert any(
        "SUBPROCESS EXIT: failed" in failure and "returncode=7" in failure
        for failure in failures
    )
    assert any(
        "SUBPROCESS TIMEOUT: sleeping" in failure for failure in failures
    )
    assert sleeping.poll() is not None
