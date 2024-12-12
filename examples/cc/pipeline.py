from dataclasses import field, dataclass
from typing import Any
import asyncio
import logging
from functools import wraps, reduce
from contextlib import contextmanager, asynccontextmanager
import pathlib

from dask.distributed import LocalCluster, Client, wait, get_worker

from waluigi.bundle import bundleclass
from waluigi.task import Task, TaskWithCleanup, ExternalTask, MemoryTask
from waluigi.runner import mk_dag, run_dag
import waluigi.runner as runner
from waluigi.target import *
from waluigi.resources import Resources

from warc_with_meta import process_and_write
from html_extractor import extract_text
from prefilter import apply_prefilter
from langfilter import apply_langid, FT

from huggingface_hub import hf_hub_download

root = pathlib.Path('data')

cc_root = pathlib.Path('/home/amaru/cc/data')
out_root = pathlib.Path('out')

def rooted(*branch):
    return LocalTarget(file=str(out_root.joinpath(*branch)))

@asynccontextmanager
async def mk_context(**kwargs):
    async with LocalCluster(n_workers=2, asynchronous=True) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            print('dashboard:', client.dashboard_link)
            context = await runner.mk_context(resources=kwargs, client=client)
            yield context

@bundleclass
class DaskTask(Task):
    async def run_async(self, context, *inputs):
        await context.client.submit(self.run, *inputs)

@bundleclass
class DaskTaskWithCleanup(TaskWithCleanup):
    async def run_async(self, context, *inputs):
        await context.client.submit(self.run, *inputs)

@bundleclass
class Raw(Task):
    branch: str
    def output(self):
        return LocalTarget(file=str(cc_root / self.branch))

    def done(self):
        assert self.output().exists(), "Raw data must exist"
        return True

@bundleclass
class ToParquet(DaskTaskWithCleanup):
    branch: str
    
    def resources(self):
        return {'worker': 1}

    def requires(self):
        return Raw(branch=self.branch)

    def output(self):
        return rooted('warc', pathlib.Path(self.branch).with_suffix('.parquet'))

    def run(self, raw):
        with self.output().tmp_path() as path:
            process_and_write(raw.path, path)

    def cleanup(self):
        self.output().path.unlink()

@bundleclass
class ToText(DaskTask):
    branch: str
    def resrouces(self):
        return {'worker': 1}

    def requires(self):
        return ToParquet(branch=self.branch)
    
    def output(self):
        return rooted('txt', pathlib.Path(self.branch).with_suffix('.parquet'))

    def run(self, txt):
        with self.output().tmp_path() as path:
            extract_text(txt.path, path)


@bundleclass
class Prefilter(DaskTask):
    branch: str

    def requires(self):
        return ToText(branch=self.branch)

    def output(self):
        return rooted('filtered', 'prefilter', pathlib.Path(self.branch).with_suffix('.parquet'))

    def run(self, txt):
        with self.output().tmp_path() as path:
            apply_prefilter(txt.path, path)


def get_glotlid(path):
    worker = get_worker()
    try:
        return worker.glotlid_model
    except AttributeError:
        worker.glotlid_model = FT(str(path))
        return worker.glotlid_model



@bundleclass
class DownloadGlotlid(Task):
    def output(self):
        return rooted('assets', 'glotlid')

    def run(self):
        with self.output().tmp_path() as path:
            hf_hub_download(repo_id='cis-lmu/glotlid', filename='model.bin', local_dir=path)

@bundleclass
class Langid(DaskTask):
    branch:str

    def resources(self):
        return {'worker': 1}
    
    def requires(self):
        return Prefilter(branch=self.branch), DownloadGlotlid()

    def output(self):
        return rooted('filtered', 'langfilter', pathlib.Path(self.branch).with_suffix('.parquet'))

    def run(self, filtered, model_dir):
        model = get_glotlid(model_dir.path / 'model.bin')
        with self.output().tmp_path() as path:
            apply_langid(filtered.path, path, model)

# Runner

async def run(dag):
    if dag:
        logging.info('Setting up cluster')
        async with mk_context(worker=4) as context:
            await run_dag(dag, context)
    else:
        logging.info('DAG empty: No tasks scheduled')

if __name__ == '__main__':
    logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            )
    tasks = []
    for file in cc_root.rglob('*.warc.gz'):
        tasks.append(Langid(branch=str(file.relative_to(cc_root))))
    logging.info('Making DAG')
    dag = mk_dag(*tasks)
    asyncio.run(run(dag))

