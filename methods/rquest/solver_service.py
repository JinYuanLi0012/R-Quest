"""Safe Solver-service handoff for sequential reuse of semantic GPUs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request


@dataclass(frozen=True)
class SolverServiceConfig:
    repo_root: Path
    model_path: str
    run_id: str
    pid_file: Path
    gpu_ids: tuple[str, ...]
    port_base: int
    health_timeout_seconds: int = 240
    release_timeout_seconds: int = 120

    @property
    def ports(self) -> tuple[int, ...]:
        return tuple(self.port_base + index for index in range(len(self.gpu_ids)))

    @classmethod
    def from_environment(cls) -> "SolverServiceConfig":
        required = {
            "VALIDITY_RZERO_REPO_ROOT": os.environ.get("VALIDITY_RZERO_REPO_ROOT"),
            "VALIDITY_RZERO_SOLVER_MODEL_PATH": os.environ.get("VALIDITY_RZERO_SOLVER_MODEL_PATH"),
            "VALIDITY_RZERO_SOLVER_RUN_ID": os.environ.get("VALIDITY_RZERO_SOLVER_RUN_ID"),
            "QUESTIONER_VLLM_PID_FILE": os.environ.get("QUESTIONER_VLLM_PID_FILE"),
            "VLLM_GPU_IDS": os.environ.get("VLLM_GPU_IDS"),
            "VLLM_PORT_BASE": os.environ.get("VLLM_PORT_BASE"),
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise RuntimeError(f"semantic service handoff missing environment: {missing}")
        return cls(
            repo_root=Path(required["VALIDITY_RZERO_REPO_ROOT"]),
            model_path=str(required["VALIDITY_RZERO_SOLVER_MODEL_PATH"]),
            run_id=str(required["VALIDITY_RZERO_SOLVER_RUN_ID"]),
            pid_file=Path(required["QUESTIONER_VLLM_PID_FILE"]),
            gpu_ids=tuple(value.strip() for value in str(required["VLLM_GPU_IDS"]).split(",") if value.strip()),
            port_base=int(str(required["VLLM_PORT_BASE"])),
            health_timeout_seconds=int(os.getenv("VALIDITY_RZERO_SERVICE_HEALTH_TIMEOUT", "240")),
            release_timeout_seconds=int(os.getenv("VALIDITY_RZERO_GPU_RELEASE_TIMEOUT", "120")),
        )


def read_pids(pid_file: Path) -> list[int]:
    if not pid_file.is_file():
        return []
    return [int(line.strip()) for line in pid_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def process_alive(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.is_file():
        try:
            # A killed process may briefly remain as a zombie until its shell
            # parent reaps it; it no longer owns a port or GPU allocation.
            if stat_path.read_text(encoding="utf-8").split()[2] == "Z":
                return False
        except (OSError, IndexError):
            pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_recorded_process_groups(pid_file: Path, timeout_seconds: int = 30) -> None:
    pids = read_pids(pid_file)
    for pid in pids:
        if process_alive(pid):
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + timeout_seconds
    while any(process_alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.25)
    for pid in pids:
        if process_alive(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if any(process_alive(pid) for pid in pids):
        raise RuntimeError(f"failed to stop recorded service processes: {pids}")
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text("", encoding="utf-8")


def port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.25)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def wait_ports_closed(ports: tuple[int, ...], timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while any(port_is_open(port) for port in ports) and time.monotonic() < deadline:
        time.sleep(0.25)
    remaining = [port for port in ports if port_is_open(port)]
    if remaining:
        raise RuntimeError(f"service ports remained open after stop: {remaining}")


def gpu_compute_pids(gpu_ids: tuple[str, ...]) -> dict[str, list[int]]:
    gpu_rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    uuid_to_index = {}
    for row in gpu_rows:
        index, uuid = (value.strip() for value in row.split(",", 1))
        uuid_to_index[uuid] = index
    result = {gpu_id: [] for gpu_id in gpu_ids}
    process_rows = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    for row in process_rows:
        if "," not in row:
            continue
        uuid, pid_text = (value.strip() for value in row.split(",", 1))
        index = uuid_to_index.get(uuid)
        if index in result:
            result[index].append(int(pid_text))
    return result


def wait_gpus_released(gpu_ids: tuple[str, ...], timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        active = gpu_compute_pids(gpu_ids)
        if not any(active.values()):
            return
        time.sleep(1)
    raise RuntimeError(f"GPU processes remained after service stop: {gpu_compute_pids(gpu_ids)}")


def service_health(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            if response.status == 200:
                return json.loads(response.read().decode("utf-8"))
    except Exception:
        pass
    return None


def wait_services_healthy(config: SolverServiceConfig) -> None:
    pids = read_pids(config.pid_file)
    if len(pids) != len(config.ports):
        raise RuntimeError(
            f"Solver PID coverage mismatch: pids={pids}, ports={config.ports}"
        )
    expected = dict(zip(config.ports, pids))
    deadline = time.monotonic() + config.health_timeout_seconds
    pending = set(config.ports)
    while pending and time.monotonic() < deadline:
        for port in list(pending):
            health = service_health(port)
            if (
                health
                and health.get("status") == "ok"
                and str(health.get("run_id")) == config.run_id
                and int(health.get("pid", -1)) == expected[port]
            ):
                pending.remove(port)
        if pending:
            time.sleep(1)
    if pending:
        observed = {port: service_health(port) for port in sorted(pending)}
        raise RuntimeError(
            "Solver services did not become healthy with the expected run/PID identity: "
            f"pending={sorted(pending)} expected={expected} observed={observed}"
        )


def start_solver_services(config: SolverServiceConfig) -> None:
    command = [
        "bash", str(config.repo_root / "vllm_service_init" / "start.sh"),
        config.model_path, config.run_id,
    ]
    subprocess.run(command, cwd=config.repo_root, env=os.environ.copy(), check=True)
    try:
        wait_services_healthy(config)
    except BaseException:
        terminate_recorded_process_groups(config.pid_file)
        raise


def stop_solver_services(config: SolverServiceConfig) -> None:
    terminate_recorded_process_groups(config.pid_file)
    wait_ports_closed(config.ports, config.release_timeout_seconds)
    wait_gpus_released(config.gpu_ids, config.release_timeout_seconds)


@contextmanager
def semantic_gpu_handoff(config: SolverServiceConfig):
    print("[rquest][novelty] stopping Solver services for frozen-base judge")
    stop_solver_services(config)
    try:
        yield
    finally:
        wait_gpus_released(config.gpu_ids, config.release_timeout_seconds)
        print("[rquest][novelty] restarting Solver services")
        start_solver_services(config)
        print("[rquest][novelty] Solver services healthy")
