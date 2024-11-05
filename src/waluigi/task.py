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
from copy import copy

@bundleclass
class Task(Bundle):
    """
    Task is the base class for tasks that are supposed to be run by waluigi. 
    To define a class, you need to define what other tasks it **requires**, 
    what it **output**s, and how to construct the output from the inputs.
    """    

    def resources(self):
        """
        **resources** define the resources needed for this task. The task
        will wait until resources are available.
        Should return a dictionary of the type {resource_name: units}
        By default returns an empty dictionary, i.e. no resources.
        """
        return {}

    def requires(self):
        """
        **requires** define the "input" tasks that this task depend on.
        The tasks **output** targets will be fed to the run_async
        method.
        """
        return []

    def _requires(self):
        """
        Wraps the requires method to make it possible to
        return a single task rather than a one element
        sequence of tasks.
        Used by the scheduler.
        """
        reqs = self.requires()
        if isinstance(reqs, Task):
            return [reqs]
        else:
            return reqs
    
    def output(self) -> Target:
        """
        **output** define the output target of this task.
        This can be a local file (LocalTarget), something
        stored in memory (MemoryTarget), or some other
        user defined target.

        The default NoTarget output means that the task
        has no output.
        """
        return NoTarget()

    def run(self, *inputs):
        """
        **run** defines the run method. This method
        is not needed if the task implements run_async.

        By default, inputs is the output targets of
        the tasks this task depends on.

        Note that if run_async is not implemented,
        this will be run by the single-threaded scheduler,
        which is, in general, not recommended.
        """
        pass

    async def noop(self):
        """
        **noop** is used by the scheduler if the task
        was *done* during scheduling.
        """
        logger.info(f'{self} already done.')
        return self

    def done(self):
        """
        **done** checks if the task has already
        finished. Used by the scheduler.
        """
        return self.output().exists()

    async def _run_after(self, context, *tasks):
        """
        **_run_after**. Used by the scheduler.
        The task wait for the input tasks
        to finish. When all input tasks
        are done, their output targets are
        passed as inputs to run_async. 
            
        All kwargs passed to the scheduler are
        passed to run_async.
        """
        if tasks:
            try:
                logger.info(f'{self} awaiting dependencies.')
                results = await asyncio.gather(*tasks)
            except Exception as e:
                logger.exception('dependency failure:')
                raise FailedDependency(self) from e
        else:
            results = []
        try:
            inputs = [x.output() for x in results]
            logger.info(f'{self} awaiting allocation.')
            async with context.resources.get_allocation(**self.resources()) as allocation:
                inner_context = copy(context)
                inner_context.allocation = allocation
                logger.info(f'{self} run started.')
                await self.run_async(inner_context, *inputs)
                logger.info(f'Run {self} done.')
            return self
        except Exception as e:
            logger.exception('run failure:')
            raise FailedRun(self) from e

    async def run_async(self, context, *inputs):
        """
        **run_async** is responsible for actually
        running the task.

        inputs: The output targets of the input tasks.
        kwargs: keyword arguments passed to the scheduler.

        By default this method just calls run with
        the inputs. However, this results in the 
        single-threaded scheduler performing the 
        work, which is not ideal.
        
        To work as intended the scheduler should
        be passed an executor (e.g. a dask Client
        or submitit Executor) that can submit work
        to be done remotely and await the results.
        """
        self.run(*inputs)

@bundleclass
class TaskWithCleanup(Task):
    """
    TaskWithCleanup should be subclassed by tasks that 
    require some kind of cleanup of its **output**.
    The cleanup is performed after this task and all 
    tasks that depend on it are done.

    This is useful for intermediate tasks whose output
    we can safely discard, or for scattering data to
    a remote cluster that should be removed when no
    longer needed.
    """    
    
    def cleanup(self):
        """
        **cleanup** defines the cleanup method. This method
        is not needed if the task implements cleanup_async.

        If the cleanup is relatively simple (e.g. unlinking
        a single file), it should be fine to do this in the single-
        threaded scheduler. 
        """
        pass

    async def _cleanup_after(self, context, *tasks):
        """
        **_cleanup_after**. Used by the scheduler.
        The cleanup waits for all tasks that depend
        on this task finish.
        
        It is passed all kwargs from the scheduler,
        and passes them in turn to clean_async.
        """
        if tasks:
       
            try:
                logger.info(f'Cleanup {self} waiting.')
                await asyncio.gather(*tasks)
            except Exception as e:
                raise FailedDependency(self) from e
        try:
            logger.info(f'Cleanup {self} entered.')
            await self.cleanup_async(context)
            logger.info(f'Cleanup {self} done.')
            return self
        except Exception as e:
            raise FailedRun(self) from e

    async def cleanup_async(self, context):
        """
        **cleanup_async** is responsible for actually
        performing the cleanup.

        kwargs: keyword arguments passed to the scheduler.

        By default this method just calls cleanup.
        This should be fine if the cleanup is simple,
        e.g. unlinking a single file. 
        
        If the cleanup is more complex or depends on some
        remote executor, it should be implement here.
        """
        self.cleanup()

@bundleclass
class ExternalTask(Task):
    """
    This task wraps a single target assumed to exist.
    Useful for tasks that depends on existing files,
    for example.
    """
    target: Target

    def output(self):
        return self.target

    def done(self):
        """
        **done** checks if the task has already
        finished. Used by the scheduler.
        For external tasks, this should always return true.
        """
        is_done = self.output().exists()
        assert is_done, f"External target does not exist: {target}"
        return True

@bundleclass
class MemoryTask(TaskWithCleanup):
    """
    This task wraps a MemoryTarget, and is intended
    to be subclassed by tasks whose outputs reside
    in memory.

    Mainly intended to be used with for example dask
    futures to keep a reference to remote objects.

    This can be useful if, for example, many tasks
    need to use a large model. This task can then
    load the model on some worker, and dependent
    tasks can get a reference to that model via
    this tasks output.
    By default the reference is deleted on cleanup.
    """
    mem: MemoryTarget = field(default_factory=MemoryTarget, compare=False)

    def output(self):
        return self.mem

    def get(self):
        return self.mem.get()

    def set(self, val):
        self.mem.set(val)

    def cleanup(self):
        self.mem.delete()
