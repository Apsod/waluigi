import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from executor import ACTIVE_STATES, SlurmExecutor, SlurmSpec
from state_target import StateTarget

import waluigi.runner as runner
from waluigi.bundle import bundleclass
from waluigi.errors import FailedRun
from waluigi.runner import mk_dag, run_dag
from waluigi.target import LocalTarget
from waluigi.task import Task

out_root = Path("out") / "slurm"


def rooted(*branch: str) -> Path:
    return out_root.joinpath(*branch)


@asynccontextmanager
async def mk_context(poll_interval: float = 30.0, **resources):
    executor = SlurmExecutor(poll_interval=poll_interval)
    context = await runner.mk_context(resources=resources, executor=executor)
    yield context


@bundleclass
class ReentrantSlurmTask(Task):
    def output(self):
        return self.state_target()

    def state_target(self) -> StateTarget:
        raise NotImplementedError

    def artifact_target(self) -> LocalTarget:
        raise NotImplementedError

    def checkpoint_target(self) -> LocalTarget | None:
        return None

    def slurm_spec(self) -> SlurmSpec:
        raise NotImplementedError

    def build_command(self, checkpoint_path: str | None) -> list[str]:
        raise NotImplementedError

    def max_attempts(self) -> int:
        return 5

    def resumable_states(self) -> set[str]:
        return {"NODE_FAIL", "PREEMPTED", "TIMEOUT"}

    def resumable_error(self, state: dict, slurm_state: str) -> bool:
        checkpoint = state.get("checkpoint")
        return slurm_state in self.resumable_states() and checkpoint is not None

    async def run_async(self, context, *inputs):
        del inputs
        state_target = self.output()
        artifact_target = self.artifact_target()
        checkpoint_target = self.checkpoint_target()
        checkpoint_path = None
        if checkpoint_target is not None:
            checkpoint_path = str(checkpoint_target.path)

        while True:
            state = state_target.read() or {}
            attempt = int(state.get("attempt", 0))
            job_id = state.get("job_id")

            if artifact_target.exists():
                state_target.mark_done(
                    artifact=str(artifact_target.path),
                    job_id=job_id,
                    attempt=attempt or None,
                )
                return

            if job_id is None:
                next_attempt = attempt + 1
                if next_attempt > self.max_attempts():
                    raise FailedRun(self)

                command = self.build_command(checkpoint_path)
                job_id = await context.executor.submit(self.slurm_spec(), command)
                state_target.mark_submitted(
                    job_id=job_id,
                    attempt=next_attempt,
                    command=command,
                    checkpoint=checkpoint_path,
                )
                await asyncio.sleep(context.executor.poll_interval)
                continue

            slurm_state = await context.executor.status(job_id)

            if slurm_state in ACTIVE_STATES:
                state_target.mark_running(job_id=job_id, attempt=attempt)
                await asyncio.sleep(context.executor.poll_interval)
                continue

            if slurm_state == "COMPLETED":
                if not artifact_target.exists():
                    state_target.mark_failed(
                        job_id=job_id,
                        attempt=attempt,
                        slurm_state=slurm_state,
                        error="Slurm job completed without producing the artifact",
                    )
                    raise FailedRun(self)
                state_target.mark_done(
                    artifact=str(artifact_target.path),
                    job_id=job_id,
                    attempt=attempt,
                )
                return

            if self.resumable_error(state, slurm_state):
                state_target.mark_checkpointed(
                    job_id=job_id,
                    attempt=attempt,
                    checkpoint=checkpoint_path,
                    slurm_state=slurm_state,
                )
                state_target.mark("pending-resubmit", job_id=None)
                continue

            state_target.mark_failed(
                job_id=job_id,
                attempt=attempt,
                slurm_state=slurm_state,
                error="Slurm job ended in a non-resumable state",
            )
            raise FailedRun(self)


@bundleclass
class CounterTask(ReentrantSlurmTask):
    branch: str = "counter"
    target_count: int = 100
    step: int = 10
    sleep_seconds: float = 5.0

    def state_target(self) -> StateTarget:
        return StateTarget(file=str(rooted(f"{self.branch}.state.json")))

    def artifact_target(self) -> LocalTarget:
        return LocalTarget(file=str(rooted(f"{self.branch}.done.json")))

    def checkpoint_target(self) -> LocalTarget:
        return LocalTarget(file=str(rooted(f"{self.branch}.checkpoint.json")))

    def slurm_spec(self) -> SlurmSpec:
        return SlurmSpec(
            job_name=f"waluigi-{self.branch}",
            time_limit="00:10:00",
            signal="TERM@30",
        )

    def build_command(self, checkpoint_path: str | None) -> list[str]:
        worker = Path(__file__).with_name("worker.py")
        command = [
            sys.executable,
            str(worker),
            "--checkpoint",
            str(self.checkpoint_target().path),
            "--output",
            str(self.artifact_target().path),
            "--target-count",
            str(self.target_count),
            "--step",
            str(self.step),
            "--sleep-seconds",
            str(self.sleep_seconds),
        ]
        if checkpoint_path is not None:
            command.extend(["--resume-from", checkpoint_path])
        return command


async def run(tasks):
    dag = mk_dag(*tasks)
    if dag:
        async with mk_context() as context:
            await run_dag(dag, context)


if __name__ == "__main__":
    asyncio.run(run([CounterTask()]))
