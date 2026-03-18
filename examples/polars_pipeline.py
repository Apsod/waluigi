import asyncio
import logging
import random
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import polars as pl

import waluigi.runner as runner
from waluigi.bundle import bundleclass
from waluigi.runner import mk_dag, run_dag
from waluigi.target import LocalTarget
from waluigi.task import Task

out_root = Path("out") / "polars"


def rooted(*branch: str) -> LocalTarget:
    return LocalTarget(file=str(out_root.joinpath(*branch)))


@bundleclass
class ProcessPoolTask(Task):
    async def run_async(self, context, *args):
        loop = asyncio.get_running_loop()
        f = self.run
        await loop.run_in_executor(context.executor, f, *args)

@bundleclass
class GenerationTask(ProcessPoolTask):
    n_rows: int = 100
    seed: int = 0

    def output(self):
        return rooted("generated.parquet")

    def run(self):
        with self.output().tmp_path() as path:
            rng = random.Random(self.seed)
            pl.DataFrame(
                {
                    "id": list(range(self.n_rows)),
                    "a": [rng.random() for _ in range(self.n_rows)],
                    "b": [rng.random() for _ in range(self.n_rows)],
                }
            ).write_parquet(path)


@bundleclass
class ClassificationTaskA(ProcessPoolTask):
    n_rows: int = 100
    seed: int = 0

    def requires(self):
        return GenerationTask(n_rows=self.n_rows, seed=self.seed)

    def output(self):
        return rooted("classification_a.parquet")

    def run(self, generated):
        with self.output().tmp_path() as path:
            pl.scan_parquet(generated.path).select(
                "id",
                (pl.col("a") > pl.col("b")).alias("a_gt_b"),
            ).sink_parquet(path)


@bundleclass
class ClassificationTaskB(ProcessPoolTask):
    n_rows: int = 100
    seed: int = 0

    def requires(self):
        return GenerationTask(n_rows=self.n_rows, seed=self.seed)

    def output(self):
        return rooted("classification_b.parquet")

    def run(self, generated):
        with self.output().tmp_path() as path:
            pl.scan_parquet(generated.path).select(
                "id",
                (pl.col("b") > 0).alias("b_gt_0"),
            ).sink_parquet(path)


@bundleclass
class JoinTask(ProcessPoolTask):
    n_rows: int = 100
    seed: int = 0

    def requires(self):
        return (
            GenerationTask(n_rows=self.n_rows, seed=self.seed),
            ClassificationTaskA(n_rows=self.n_rows, seed=self.seed),
            ClassificationTaskB(n_rows=self.n_rows, seed=self.seed),
        )

    def output(self):
        return rooted("joined_filtered.parquet")

    def run(self, generated, classified_a, classified_b):
        with self.output().tmp_path() as path:
            (
                pl.scan_parquet(generated.path)
                .join(pl.scan_parquet(classified_a.path), on="id")
                .join(pl.scan_parquet(classified_b.path), on="id")
                .filter(pl.col("a_gt_b") & pl.col("b_gt_0"))
                .select("id", "a", "b")
                .sink_parquet(path)
            )


async def run(tasks):
    dag = mk_dag(*tasks)
    if dag:
        async with mk_context() as context:
            await run_dag(dag, context)
    else:
        logging.info("DAG empty: No tasks scheduled")


@asynccontextmanager
async def mk_context(max_workers=4, **resources):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        context = await runner.mk_context(resources=resources, executor=executor)
        yield context


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(run([JoinTask(n_rows=1000, seed=42)]))
