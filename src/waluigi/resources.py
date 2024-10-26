import copy

from collections import Counter
from contextlib import asynccontextmanager
from asyncio import Condition
from dataclasses import dataclass

from waluigi import logger
from waluigi.errors import *


def as_ctr(*args, **kwargs):
    match args, kwargs:
        case [Counter() as ctr], {}:
            pass
        case [], counts:
            ctr = Counter(**counts)
        case _:
            raise ValueError("Type failure")
    return ctr


class Resources:
    def __init__(self, *args, **kwargs):
        self.resources = as_ctr(*args, **kwargs)
        self.state = copy.deepcopy(self.resources)
        self.cond = Condition()

    @classmethod
    async def init(cls, *args, **kwargs):
        ret = cls(*args, **kwargs)
        await ret.notify()
        return ret


    async def notify(self):
        async with self.cond:
            self.cond.notify_all()

    async def add_resources(self, *args, **kwargs):
        resources = as_ctr(*args, **kwargs)
        if resources:
            self.state += resources
            assert self.state <= self.resources
            await self.notify()
    
    @asynccontextmanager
    async def wait_for_resources(self, *args, **kwargs):
        requirement = as_ctr(*args, **kwargs)
        if not (requirement < self.resources): 
            raise ResourceError(f"requested more than maximum resources: {requirement} </= {self.resources}")
        async with self.cond:
            await self.cond.wait_for(lambda: requirement <= self.state)
            self.state -= requirement
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
        await self.release(**self.acquired)

    async def release(self, *args, **kwargs):
        part = as_ctr(*args, **kwargs)
        assert part <= self.acquired
        self.acquired -= part
        await self.supply.add_resources(part)
