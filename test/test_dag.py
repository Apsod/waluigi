import sys
import tempfile
import unittest
from dataclasses import field
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waluigi.bundle import bundleclass
from waluigi.graph import Graph
from waluigi.runner import mk_context, mk_dag, run_dag
from waluigi.target import LocalTarget, MemoryTarget
from waluigi.task import ExternalTask, Task, TaskWithCleanup


@bundleclass
class CycleA(Task):
    def requires(self):
        return CycleB()


@bundleclass
class CycleB(Task):
    def requires(self):
        return CycleA()


@bundleclass
class ValueTask(Task):
    name: str
    mem: MemoryTarget = field(default_factory=MemoryTarget, compare=False)

    def output(self):
        return self.mem

    def run(self):
        self.output().set(self.name)


@bundleclass
class OrderedConsumer(Task):
    reverse: bool = False
    mem: MemoryTarget = field(default_factory=MemoryTarget, compare=False)

    def requires(self):
        left = ValueTask(name="left")
        right = ValueTask(name="right")
        return (right, left) if self.reverse else (left, right)

    def output(self):
        return self.mem

    def run(self, *inputs):
        self.output().set(tuple(target.get() for target in inputs))


@bundleclass
class Producer(TaskWithCleanup):
    mem: MemoryTarget = field(default_factory=MemoryTarget, compare=False)

    def output(self):
        return self.mem

    def run(self):
        EVENTS.append("produce")
        self.output().set("payload")

    def cleanup(self):
        EVENTS.append("cleanup")
        self.output().delete()


@bundleclass
class Consumer(Task):
    mem: MemoryTarget = field(default_factory=MemoryTarget, compare=False)

    def requires(self):
        return Producer()

    def output(self):
        return self.mem

    def run(self, producer):
        EVENTS.append(f"consume:{producer.get()}")
        self.output().set("done")


EVENTS: List[str] = []


class GraphRegressionTests(unittest.TestCase):
    def test_get_preserves_dependency_insertion_order(self):
        graph = Graph()
        graph.add("root")
        graph.add("left", "root")
        graph.add("right", "root")

        self.assertEqual(list(graph.get("root").left), ["left", "right"])


class RunnerRegressionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        EVENTS.clear()

    def test_mk_dag_rejects_cycles_without_recursing_forever(self):
        with self.assertRaisesRegex(ValueError, "Cycles detected"):
            mk_dag(CycleA())

    async def test_run_dag_uses_frozen_dependency_order(self):
        task = OrderedConsumer(reverse=False)
        dag = mk_dag(task)

        # Mutate the task after DAG construction. run_dag should still use the
        # dependency order captured in the DAG, not a fresh requires() result.
        object.__setattr__(task, "reverse", True)

        context = await mk_context()
        await run_dag(dag, context)

        self.assertEqual(task.output().get(), ("left", "right"))

    def test_external_task_reports_missing_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing.txt"
            task = ExternalTask(target=LocalTarget(file=str(path)))
            with self.assertRaisesRegex(AssertionError, str(path)):
                task.done()

    async def test_cleanup_runs_after_dependents_finish(self):
        task = Consumer()
        dag = mk_dag(task)
        context = await mk_context()

        await run_dag(dag, context)

        self.assertEqual(EVENTS, ["produce", "consume:payload", "cleanup"])
