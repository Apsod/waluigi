# Waluigi

## What is this

Waluigi is a package for building complex pipelines of batch jobs. 
It is heavily inspired by [Luigi](https://github.com/spotify/luigi),
with some differences:

- Waluigi is designed around pythons asyncio framework.
- Waluigi has no concept of workers: The scheduler runs
in a single (async) thread. The idea is that users supply
an executor (e.g. dask, submitit) that does the work remotely.
- Waluigi is meant to be run completely from within python.
It does not have luigis commandline functionality.
- Waluigi only implements a small subset of luigi functionality. 

## Tasks: `waluigi.task`

Tasks are the things we wish to run. They are defined by the tasks they `require`, 
and the target they `output`. To actually output anything of value they also need 
to implement a *run_async* or *run* method that materializes the output.

```
@bundleclass
class UpperCase(Task):
    file: str

    def requires(self):
        return External(target=LocalTarget(self.file))

    def output(self):
        return LocalTarget(f'{self.file}_uppercased')

    def run(self, input):
        with (
            input.open() as f_in,
            self.output().open('wt') as f_out
            ):
            f_out.write(f_in.read().upper())
```

The above tasks reads a file, uppercases it, and writes the result to `$file_uppercased`.

For tasks whose outputs we don't want to persist after their downstreams tasks are finished,
we supply TaskWithCleanup. This task requires an additional *cleanup* or *cleanup_async* method.


```
@bundleclass
class UpperCase(TaskWithCleanup):
    file: str
    
    ...
    
    def cleanup(self):
        self.output().path.unlink()
```

## Targets: `waluigi.target`


Targets are the things that tasks output. The only thing a target needs to implement
is an `exists` method, which tells the scheduler if the task needs to be run or not.

Waluigi comes with three predefined targets:
`LocalTarget`, `MemoryTarget`, and `NoTarget`. 

LocalTarget points to a local file, and comes with convenient context managers such as `open`
and `tmp_path`. During writing, LocalTarget writes to a temporary file using the `tmp_path` 
context manager to avoid failed tasks to result in "existing" output. The temporary file is 
then moved to the desired path on exit.

MemoryTarget is a target that resides in scheduler memory. This is mainly intended to be
used for references to remote data. One use case is to load a large model remotely as a 
separate task, and have several tasks use the loaded model (leaving data shuffling to 
the executor). 

NoTarget is a target that never exists. It is mainly intended for debugging purposes.

## Runner: `waluigi.runner`

The runner is responsible for constructing the graph of task dependencies and create
asyncio tasks that perform the actual work. The two main methods are `mk_dag` and `run_dag`. 

`mk_dag` takes a lists of tasks and constructs a graph by iterating over tasks dependencies.
This graph is then topologically sorted and tasks are returned along with their dependencies
and dependents. 

`run_dag` takes this dag and schedules all runs, along with cleanup methods, using asyncio.
`run_dag` also takes arbitrary keyword arguments. These are sent to each tasks *run_async*
method and used to supply the Tasks with remote executors, resource limitation, et.c. 

## Example:

```
import os
import asyncio

import polars
from dask.distributed import LocalCluster, Client

from waluigi.runner import *
from waluigi.task import *
from waluigi.target import *

class DaskTask(Task):
    def run_async(*inputs, client=client, **kwargs):
        await client.submit(self.run(*inputs))

root = 'path/to/data/folder'

class Raw(External):
    branch: str
    def output(self):
        return LocalTarget(file=os.path.join(root, 'raw', self.branch))

class QualitySignals(DaskTask):
    branch: str
    def requires(self):
        return Raw(branch=self.branch)

    def output(self):
        return LocalTarget(file=os.path.join(root, 'signals', self.branch))

    def run(self, infile):
        with self.output().tmp_path() as path:
            polars.read_parquet(infile).select(...).write_parquet(path)

class Filtered(DaskTask):
    branch: str
    def requires(self):
        return Raw(branch=self.branch), QualitySignals(branch=self.branch)

    def output(self):
        return LocalTarget(file=os.path.join(root, 'filtered', self.branch))

    def run(self, raw, signals):
        with self.output().tmp_path() as path:
            raw_df = polars.read_parquet(raw.path)
            signals_df = polars.read_parquet(signals.path)
            joined = polars.concat([raw_df, signals_df]).select(...)
            joined.write_parquet(path)



async def run(dag):
    async with LocalCluster(asynchronous=True) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            await run_dag(dag, client=client)

if __name__ == '__main__':
    tasks = [Filtered(branch='xaa.parquet'), ...] # List of tasks we want to run
    dag = mk_dag(*tasks)
    asyncio.run(run(dag))
```
