import asyncio
import shlex
from dataclasses import dataclass, field

ACTIVE_STATES = {
    "BOOT_FAIL",
    "COMPLETING",
    "CONFIGURING",
    "PENDING",
    "RUNNING",
    "SUSPENDED",
}

TERMINAL_STATES = {
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}


@dataclass(frozen=True)
class SlurmSpec:
    job_name: str
    time_limit: str
    partition: str | None = None
    cpus_per_task: int | None = None
    memory: str | None = None
    signal: str | None = "TERM@60"
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def sbatch_args(self) -> list[str]:
        args = [
            "--parsable",
            "--job-name",
            self.job_name,
            "--time",
            self.time_limit,
        ]
        if self.partition is not None:
            args.extend(["--partition", self.partition])
        if self.cpus_per_task is not None:
            args.extend(["--cpus-per-task", str(self.cpus_per_task)])
        if self.memory is not None:
            args.extend(["--mem", self.memory])
        if self.signal is not None:
            args.extend(["--signal", self.signal])
        args.extend(self.extra_args)
        return args


class SlurmExecutor:
    def __init__(self, poll_interval: float = 30.0):
        self.poll_interval = poll_interval

    async def _run_command(self, *args: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode().strip() or "Slurm command failed")
        return stdout.decode().strip()

    async def submit(self, spec: SlurmSpec, command: list[str]) -> str:
        stdout = await self._run_command(
            "sbatch",
            *spec.sbatch_args(),
            "--wrap",
            shlex.join(command),
        )
        return stdout.split(";", 1)[0].strip()

    async def status(self, job_id: str) -> str:
        stdout = await self._run_command(
            "sacct",
            "-n",
            "-P",
            "-j",
            job_id,
            "--format=State",
        )
        for line in stdout.splitlines():
            line = line.strip()
            if line:
                return self.normalize_state(line.split("|", 1)[0])
        return "UNKNOWN"

    async def wait(self, job_id: str) -> str:
        while True:
            status = await self.status(job_id)
            if status not in ACTIVE_STATES:
                return status
            await asyncio.sleep(self.poll_interval)

    async def cancel(self, job_id: str) -> None:
        await self._run_command("scancel", job_id)

    @staticmethod
    def normalize_state(state: str) -> str:
        return state.split()[0].split("+")[0].upper()
