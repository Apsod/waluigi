from dataclasses import field
import asyncio
import logging

from dask.distributed import LocalCluster, Client

from waluigi.bundle import bundleclass
from waluigi.task import Task, TaskWithCleanup, ExternalTask, MemoryTask
from waluigi.runner import mk_dag, run_dag
from waluigi.target import *
from waluigi.resources import Resources

import polars as pl
import pathlib

root = pathlib.Path('/path/to/some/parquet/dir')

def rooted(*branch):
    return LocalTarget(file=str(root.joinpath(*branch)))

@bundleclass
class Head(TaskWithCleanup):
    branch: str

    def requires(self):
        return ExternalTask(target=rooted('raw', self.branch))

    def output(self):
        return rooted('head', self.branch)

    def run(self, raw):
        with self.output().tmp_path() as path:
            pl.scan_parquet(raw.file).head().sink_parquet(path)

    def cleanup(self):
        self.output().path.unlink()

    async def run_async(self, raw, resources, client):
        async with resources.wait_for_resources(head=1):
            await client.submit(self.run, raw)

@bundleclass
class LoadStuff(MemoryTask):
    async def run_async(self, client, **kwargs):
        fut = await client.submit(lambda: list(range(100)))
        self.mem.set(fut)

@bundleclass
class PrintStuff(Task):
    ix: int

    def requires(self):
        return LoadStuff()

    def run(self, xs):
        print('Print:', xs[self.ix])

    async def run_async(self, mem, client, **kwargs):
        result = await client.submit(self.run, mem.get())

@bundleclass
class Upper(TaskWithCleanup):
    branch: str

    def requires(self):
        return Head(branch=self.branch)

    def output(self):
        return rooted('upper', self.branch)

    def run(self, head):
        with self.output().tmp_path() as path:
            pl.scan_parquet(head.file).select(pl.col('text').str.to_uppercase()).sink_parquet(path)

    def cleanup(self):
        self.output().path.unlink()

    async def run_async(self, head, resources, client):
        async with resources.wait_for_resources(upper=2) as supply:
            await supply.release(upper=1)
            await client.submit(self.run, head)

@bundleclass
class Join(Task):
    branch: str

    def requires(self):
        return Head(branch=self.branch), Upper(branch=self.branch)

    def output(self):
        return rooted('joined', self.branch)

    def run(self, head, upper):
        head = pl.scan_parquet(head.file)
        upper = pl.scan_parquet(upper.file).select(pl.col('text').alias('upper'))

        with self.output().tmp_path() as path:
            pl.concat([head, upper], how='horizontal').collect().write_parquet(path)

    async def run_async(self, head, upper, resources, client):
        await client.submit(self.run, head, upper)

async def run(dag):
    if dag:
        logging.info('Setting up cluster')
        resources = await Resources.init(upper=3, head=5)
        async with LocalCluster(asynchronous=True) as cluster:
            async with Client(cluster, asynchronous=True) as client:
                print(client.dashboard_link)
                await run_dag(dag, resources=resources, client=client)
                await asyncio.sleep(5)
    else:
        logging.info('DAG empty: No tasks scheduled')

if __name__ == '__main__':
    logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            )
    rawdir = root / 'raw'
    logging.info('Creating tasks')
    tasks = []
    for file in rawdir.rglob('*.parquet'):
        tasks.append(Join(branch=str(file.relative_to(rawdir))))
    tasks.append(PrintStuff(ix=2))
    tasks.append(PrintStuff(ix=3))
    tasks.append(PrintStuff(ix=84))
    logging.info('Making DAG')
    dag = mk_dag(*tasks)
    asyncio.run(run(dag))
