from dataclasses import field, dataclass
from typing import Any
import asyncio
import logging
from functools import wraps
from contextlib import contextmanager
import pathlib

from dask.distributed import LocalCluster, Client
import polars as pl
import polars.datatypes as pld

from waluigi.bundle import bundleclass
from waluigi.task import Task, TaskWithCleanup, ExternalTask, MemoryTask
from waluigi.runner import mk_dag, run_dag
from waluigi.target import *
from waluigi.resources import Resources

root = pathlib.Path('data')

def rooted(*branch):
    return LocalTarget(file=str(root.joinpath(*branch)))

@bundleclass
class DaskTask(Task):
    async def run_async(self, *inputs, client, **kwargs):
        await client.submit(self.run, *inputs)

@bundleclass
class Raw(Task):
    branch: str
    def output(self):
        return rooted('raw', self.branch)

    def done(self):
        assert self.output().exists(), "Raw data must exist"
        return True

def alias(f):
    name = f.__name__
    @wraps(f)
    def inner(*args, **kwargs):
        return f(*args, **kwargs).alias(name)
    return inner

def ratio(a, b):
    return (a / b).cast(pld.Float32)

def ratio_short_lines():
    text = pl.col('text')
    line_lengths = text.str.split('\n').list.eval(pl.element().str.strip_chars().str.len_chars())
    total = line_lengths.list.len()
    empty = line_lengths.list.eval(pl.element() == 0).list.sum()
    short = line_lengths.list.eval(pl.element() <= 30).list.sum()
    return ratio(short - empty, total - empty)

def ratio_alphanumeric():
    text = pl.col('text')
    alphanumeric = text.str.count_matches(r'\w')
    total = text.str.len_chars()
    return ratio(alphanumeric, total)

def apply_signals(df, *functions):
    return df.select(*[alias(f)() for f in functions])

@bundleclass
class QualitySignals(DaskTask):
    branch: str

    def requires(self):
        return Raw(branch=self.branch)

    def output(self):
        return rooted('signals', self.branch)

    def run(self, raw):
        with self.output().tmp_path() as path:
            df = pl.scan_parquet(raw.path)
            out = apply_signals(df, ratio_short_lines, ratio_alphanumeric)
            out.sink_parquet(path)

@dataclass
class Bound:
    low: Any = None
    high: Any = None
    def __call__(self, x):
        match self.low, self.high:
            case (None, None):
                return True
            case (None, high):
                return x <= high
            case (low, None):
                return low <= x
            case (low, high):
                return x.is_between(low, high)

FILTER = dict(
    ratio_short_lines=Bound(high=0.3),
    ratio_alphanumeric=Bound(low=0.8),
)

def mk_filter(**kwargs):
    return reduce(lambda a, b: a & b, [f(pl.col(k)) for k, f in kwargs.items()])


@bundleclass
class Filter(DaskTask):
    branch: str

    def requires(self):
        return Raw(branch=self.branch), QualitySignals(branch=self.branch)

    def output(self):
        return rooted('filtered', self.branch)

    def run(self, raw, signals):
        with self.output().tmp_path() as path:
            raw = pl.read_parquet(raw.path)
            signals = pl.read_parquet(signals.path)
            pl.concat([raw, signals], how='horizontal').filter(((pl.col('ratio_alphanumeric') >=.8) & (pl.col('ratio_short_lines') <= 0.1))).select(raw.columns).write_parquet(path)


# Runner

async def run(dag):
    if dag:
        logging.info('Setting up cluster')
        resources = await Resources.init(A=1, upper=3, head=5)
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
    tasks = []
    for file in rawdir.rglob('*.parquet'):
        tasks.append(Filter(branch=str(file.relative_to(rawdir))))
    logging.info('Making DAG')
    dag = mk_dag(*tasks)
    asyncio.run(run(dag))
