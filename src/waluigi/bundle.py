from dataclasses import dataclass, asdict, fields
from functools import partial
from pydoc import locate
import json

bundleclass = partial(dataclass, frozen=True, eq=True, kw_only=True)

PREFIX = '__bundle_class.'

def from_dict(value):
    if isinstance(value, dict) and len(value) == 1 and list(value)[0].startswith(PREFIX):
        (key, val), = value.items()
        clspath = key[len(PREFIX):]
        return locate(clspath)(**{k: from_dict(v) for k, v in val.items()})
    return value

def from_json(json_str):
    return from_dict(json.loads(json_str))

    
@bundleclass
class Bundle:
    def _asdict(self):
        cls = self.__class__
        args = {}
        for f in fields(self):
            key = f.name
            val = getattr(self, f.name)
            if isinstance(val, Bundle):
                args[key] = val._asdict()
            else:
                args[key] = val
        name = '.'.join((cls.__module__, cls.__qualname__))
        return {PREFIX + name: args}

    def tojson(self):
        return json.dumps(self._asdict())
