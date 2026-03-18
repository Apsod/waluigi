import json
from datetime import datetime, timezone
from pathlib import Path

from waluigi.bundle import bundleclass
from waluigi.target import LocalTarget, Target


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@bundleclass
class StateTarget(Target):
    file: str

    @property
    def path(self) -> Path:
        return Path(self.file)

    def exists(self) -> bool:
        state = self.read()
        if not state or state.get("status") != "done":
            return False

        artifact = state.get("artifact")
        return artifact is None or Path(artifact).exists()

    def storage(self) -> LocalTarget:
        return LocalTarget(file=self.file)

    def read(self) -> dict | None:
        if not self.storage().exists():
            return None
        with self.storage().open() as handle:
            return json.load(handle)

    def write(self, state: dict) -> None:
        payload = dict(state)
        payload["updated_at"] = utc_now()
        with self.storage().open("wt") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def mark(self, status: str, **fields) -> dict:
        state = self.read() or {}
        state.update(fields)
        state["status"] = status
        self.write(state)
        return state

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def mark_submitted(
        self,
        *,
        job_id: str,
        attempt: int,
        command: list[str],
        checkpoint: str | None,
    ) -> dict:
        return self.mark(
            "submitted",
            job_id=job_id,
            attempt=attempt,
            command=command,
            checkpoint=checkpoint,
        )

    def mark_running(self, *, job_id: str, attempt: int) -> dict:
        return self.mark("running", job_id=job_id, attempt=attempt)

    def mark_checkpointed(
        self,
        *,
        job_id: str,
        attempt: int,
        checkpoint: str | None,
        slurm_state: str,
    ) -> dict:
        return self.mark(
            "checkpointed",
            job_id=job_id,
            attempt=attempt,
            checkpoint=checkpoint,
            slurm_state=slurm_state,
        )

    def mark_done(
        self,
        *,
        artifact: str | None,
        job_id: str | None,
        attempt: int | None,
    ) -> dict:
        return self.mark(
            "done",
            artifact=artifact,
            job_id=job_id,
            attempt=attempt,
        )

    def mark_failed(
        self,
        *,
        job_id: str | None,
        attempt: int | None,
        slurm_state: str,
        error: str,
    ) -> dict:
        return self.mark(
            "failed",
            job_id=job_id,
            attempt=attempt,
            slurm_state=slurm_state,
            error=error,
        )

    def artifact_target(self) -> LocalTarget | None:
        state = self.read() or {}
        artifact = state.get("artifact")
        if artifact is None:
            return None
        return LocalTarget(file=artifact)
