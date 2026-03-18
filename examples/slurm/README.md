# Slurm Reentry Example

This folder shows one way to model long-running, reentrant Slurm jobs in Waluigi.

The key idea is that the task output is a `StateTarget`, not the final artifact itself.
The state file is the authoritative completion record and is only considered done when:

- `status == "done"`
- the referenced artifact exists, if one is recorded

## Files

- `state_target.py`: durable state target with helpers such as `mark_submitted`, `mark_done`, and `mark_failed`
- `executor.py`: lightweight async wrapper around `sbatch`, `sacct`, and `scancel`
- `task.py`: `ReentrantSlurmTask`, a base class that polls, checkpoints, and resubmits
- `worker.py`: resumable example worker that checkpoints progress to JSON and writes a final artifact on completion

## How It Works

`ReentrantSlurmTask.run_async()` treats the state file as the task output and drives a small state machine:

1. If the final artifact already exists, mark the state target `done`.
2. If no job is active, submit a new Slurm job and record its job id in the state file.
3. Poll Slurm until the job reaches a terminal state.
4. If the job completed and the artifact exists, mark `done`.
5. If the job timed out or was preempted and a checkpoint exists, mark the state as checkpointed and resubmit.
6. Otherwise mark the task as failed.

Downstream tasks receive the `StateTarget` and can resolve the real artifact through `artifact_target()`.
