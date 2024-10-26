from pathlib import Path
from uuid import uuid4
from contextlib import contextmanager
from waluigi.bundle import *
from waluigi import logger
from dataclasses import field

@bundleclass
class Target(Bundle):
    def exists(self) -> bool:
        pass

@bundleclass
class NoTarget(Bundle):
    def exists(self) -> bool:
        return False

class Wrapped(object):
    def __init__(self):
        pass

    def set(self, val):
        if hasattr(self, 'val'):
            raise AttributeError('Trying to set wrapped object twice')
        self.val = val

    def get(self):
        try:
            return self.val
        except AttributeError:
            raise AttributeError('Trying to get wrapped object without setting it first')

    def delete(self):
        try:
            del self.val
        except AttributeError:
            raise AttributeError('Trying to delete unset wrapped object')

@bundleclass
class MemoryTarget(Bundle):
    val: Wrapped = field(default_factory=Wrapped)
    def exists(self) -> bool:
        return False

    def set(self, val):
        self.val.set(val)

    def get(self):
        return self.val.get()

    def delete(self):
        self.val.delete()

@bundleclass
class LocalTarget(Bundle):
    file: str
    force: bool = False

    @property
    def path(self):
        return Path(self.file)

    def exists(self):
        return self.path.exists() and (not self.force)

    @contextmanager
    def open(self, mode='r'):
        if mode.startswith('r'):
            with open(self.path, mode) as handle:
                yield handle
        elif mode.startswith('w'):
            with self.tmp_path() as path:
                with open(path, mode) as handle:
                    yield handle

    @contextmanager
    def tmp_path(self):
        rid = uuid4()
        tmp_path = Path(f'{self.file}-TMP-{rid}')
        tmp_path.parent.mkdir(exist_ok=True, parents=True)
        try:
            yield tmp_path
            tmp_path.rename(self.file)
        finally:
            if tmp_path.exists():
                tmp_path.rename(f'{self.file}-FAILED-{rid}')
