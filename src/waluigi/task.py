from dataclasses import dataclass
from functools import partial, wraps
import json
import random
from dataclasses import *

from waluigi import logger
from waluigi.graph import *
from waluigi.bundle import *
from waluigi.target import * 
from waluigi.errors import *

import asyncio

@bundleclass
class Task(Bundle):

    def requires(self):
        return []

    def _requires(self):
        reqs = self.requires()
        if isinstance(reqs, Task):
            return [reqs]
        else:
            return reqs
    
    def output(self) -> Target:
        return NoTarget()

    def run(self, *inputs):
        pass

    async def noop(self):
        logger.info(f'{self} already done.')
        return self

    def done(self):
        return self.output().exists()

    async def _run_after(self, *tasks, **kwargs):
        if tasks:
            try:
                logger.info(f'Run {self} waiting.')
                results = await asyncio.gather(*tasks)
            except Exception as e:
                logger.exception('dependency failure:')
                raise FailedDependency(self) from e
        else:
            results = []
        try:
            inputs = [x.output() for x in results]
            logger.info(f'Run {self} entered.')
            await self.run_async(*inputs, **kwargs)
            logger.info(f'Run {self} done.')
            return self
        except Exception as e:
            logger.exception('run failure:')
            raise FailedRun(self) from e

    async def run_async(self, *inputs, **kwargs):
        self.run(*inputs)

@bundleclass
class TaskWithCleanup(Task):
    def cleanup(self):
        pass

    async def _cleanup_after(self, *tasks, **kwargs):
        if tasks:
            try:
                logger.info(f'Cleanup {self} waiting.')
                await asyncio.gather(*tasks)
            except Exception as e:
                raise FailedDependency(self) from e
        try:
            logger.info(f'Cleanup {self} entered.')
            await self.cleanup_async(**kwargs)
            logger.info(f'Cleanup {self} done.')
            return self
        except Exception as e:
            raise FailedRun(self) from e

    async def cleanup_async(self, **kwargs):
        self.cleanup()

@bundleclass
class ExternalTask(Task):
    target: Target

    def output(self):
        return self.target

@bundleclass
class MemoryTask(TaskWithCleanup):
    mem: MemoryTarget = field(default_factory=MemoryTarget, compare=False)

    def output(self):
        return self.mem

    def get(self):
        return self.mem.get()

    def set(self, val):
        self.mem.set(val)

    def cleanup(self):
        self.mem.delete()
