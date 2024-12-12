from pathlib import Path
from uuid import uuid4
from contextlib import contextmanager, nullcontext
from waluigi.bundle import *
from waluigi import logger
from dataclasses import field

@bundleclass
class Target(Bundle):
    """
    A target class. Needs to implement exists.
    """
    def exists(self) -> bool:
        """
        exists checks if the target exists.
        It is up to the user to define appropriate
        an appropriate method. For local files, see
        MemoryTarget.

        Note that exists is only used by the scheduler
        when scheduling tasks. The output targets of 
        finished tasks are assumed to exist.
        """
        pass

@bundleclass
class NoTarget(Target):
    """
    A dummy target that never exists.
    Mainly useful for debugging and tasks
    that only have side-effects.
    """
    def exists(self) -> bool:
        return False

class Wrapped(object):
    """
    A wrapper around a value, used by memory-targets.
    """
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
class MemoryTarget(Target):
    """
    A target that resides in memory.
    """
    val: Wrapped = field(default_factory=Wrapped)
    def exists(self) -> bool:
        return False

    def set(self, val):
        """
        set the underlying value to val
        raises an AttributeError if the value is already set
        """
        self.val.set(val)

    def get(self):
        """
        gets the underlying value
        raises an AttributeError if the value is not set
        """
        return self.val.get()

    def delete(self):
        """
        deletes the underlying value, allowing it to be garbage collected.
        raises an AttributeError if the value is not set
        """
        self.val.delete()

@bundleclass
class LocalTarget(Target):
    """
    A target that points to a local file.
    Uses a temporary file during writing to
    ensure that the output of aborted tasks 
    are not treated as done on subsequent runs.

    file: path to file
    force: if force, this target will never be 
    considered as existing. i.e. the corresponding
    task will be run and the old file overwritten.
    """
    file: str
    force: bool = False

    @property
    def path(self):
        """
        returns self.file as a pathlib.Path.
        """
        return Path(self.file)

    def exists(self):
        return self.path.exists() and (not self.force)

    @contextmanager
    def open(self, mode='r'):
        """
        Get a handle to the file. 
        If writing, the handle points to a temporary file
        that is moved to self.file on exit (see LocalTarget.tmp_path)
        """
        if mode.startswith('r'):
            path_context = nullcontext(self.path)
        elif mode.startswith('w'):
            path_context = self.tmp_path()
        else:
            raise ValueError(f'{mode} not a valid open mode')

        with (
                path_context as path,
                open(path, mode) as handle,
                ):
            yield handle

    @contextmanager
    def tmp_path(self):
        """
        Context manager that yields a temporary path.
        The temporary path is moved to self.file on exit.
        """
        rid = uuid4()
        tmp_path = Path(f'{self.file}-TMP-{rid}')
        tmp_path.parent.mkdir(exist_ok=True, parents=True)
        try:
            yield tmp_path
            tmp_path.rename(self.file)
        finally:
            if tmp_path.exists():
                tmp_path.rename(f'{self.file}-FAILED-{rid}')
