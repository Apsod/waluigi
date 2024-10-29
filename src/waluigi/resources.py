import copy

from collections import Counter
from contextlib import asynccontextmanager
from asyncio import Condition
from dataclasses import dataclass

from waluigi import logger
from waluigi.errors import *


def as_ctr(*args, **kwargs):
    """
    Helper function to construct Counter from *args, **kwargs.
    Takes either a Counter as input, or kwargs with counts.
    i.e. either `as_ctr(Counter(A=2,B=3))` or `as_ctr(A=2, B=3)`
    """
    match args, kwargs:
        case [Counter() as ctr], {}:
            pass
        case [], counts:
            ctr = Counter(**counts)
        case _:
            raise ValueError("Type failure")
    return ctr


class Resources:
    """
    Resources is used to control limited resources used
    by tasks. This is only used to block tasks from
    running, and does not interact with an executor.
    Still, it can be used to limit the number of concurrent
    running slurm jobs with submitit or memory-heavy tasks in dask.

    Initialize using e.g. `await Resource.init(gpu=8)` and 
    use `the get_allocation` context manager:
    ```
    GPUTask(Task):
        def run_asyn(self, *inputs, resources):
            with resources.get_allocation(gpu=2) as allocation:
                ... do heavy work
                allocation.release(gpu=1) 
                .. do less heavy work
            ...
    """
    def __init__(self, *args, **kwargs):
        self.used = Counter()
        self.available = as_ctr(*args, **kwargs)
        self.cond = Condition()

    @classmethod
    async def init(cls, *args, **kwargs):
        """
        Initiliazie the Resources. 
        Can either be called with a counter or with kwargs:
        ```
        Resources.init(Counter(A=2, B=3))
        Resources.init(A=2, B=3)
        ```
        """
        ret = cls(*args, **kwargs)
        await ret.notify()
        return ret
    
    def total(self):
        return self.used + self.available

    async def notify(self):
        async with self.cond:
            self.cond.notify_all()

    async def return_resources(self, *args, **kwargs):
        """
        Return resources to the supply.
        """
        resources = as_ctr(*args, **kwargs)
        if not (resources <= self.used):
            raise ResourceError('Returning resources not in use: {resources} </= {self.used}')
        if resources:
            self.used -= resources
            self.available += resources
            await self.notify()

    async def add_resources(self, *args, **kwargs):
        """
        Add resources to the supply. (increasing total resources)
        """
        resources = as_ctr(*args, **kwargs)
        if resources:
            self.available += resources
            await self.notify()

    async def request_resources(self, *args, **kwargs):
        requirement = as_ctr(*args, **kwargs)
        if not (requirement <= self.total()):
            raise ResourceError(f"Requested incompatible resources: {requirement} </= {self.total()}")
        async with self.cond:
            await self.cond.wait_for(lambda: requirement <= self.available)
            self.used += requirement
            self.available -= requirement
        return requirement
    
    @asynccontextmanager
    async def get_allocation(self, *args, **kwargs):
        """
        Contextmanager that create an `Allocation` which 
        can be used to release partial resources or request more.
        Releases all allocated resources back to the supply on exit.

        Raises ResourceError if requested resources <= total resources.
        """
        requirement = await self.request_resources(*args, **kwargs)
        allocation = Allocation(requirement, self)
        try:
            yield allocation
        finally:
            await allocation.release_all()

@dataclass
class Allocation:
    acquired: Counter
    supply: Resources

    async def release_all(self):
        """
        Release all resources in this allocation back to the supply.
        """
        await self.release(**self.acquired)

    async def request(self, *args, **kwargs):
        """
        Request more resources. Using this can lead to deadlocks.
        e.g.
        ```
        resources = await Resources.init(A=2)
        ....
        # Task 1:
        with resources.get_allocation(A=1) as alloc:
            ...
            await alloc.request(A=1)

        # Task 2:
        with resources.get_allocation(A=1) as alloc:
            ..
            await alloc.request(A=1)
        ```

        Raises ResourceError if requesting more resources than the
        total available resources.
        """
        req = await self.supply.request_resources(*args, **kwargs)
        self.acquired += req

    async def release(self, *args, **kwargs):
        """
        Release resources. Can be used to release partial resources
        during allocation:
        ```
        with resources.get_allocation(A=3, B=2) as alloc:
            ...
            alloc.release(A=1, B=1) # <- release {A=1, B=1} back to the supply.
            ...
        """
        part = as_ctr(*args, **kwargs)
        assert part <= self.acquired
        self.acquired -= part
        await self.supply.return_resources(part)
