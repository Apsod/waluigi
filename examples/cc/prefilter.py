from dataclasses import dataclass
from typing import Any
import functools

import polars as pl
import polars.datatypes as pld

@dataclass
class Bound:
    low: Any = None
    high: Any = None

    def __call__(self, x):
        match self.low, self.high:
            case (None, None):
                return True
            case (None, high):
                return x <= high
            case (low, None):
                return low <= x
            case (low, high):
                return x.is_between(low, high)

def ratio(a, b):
    return (a / b).cast(pld.Float32)

def num_latin_chars():
    return pl.col('text').str.count_matches(r'\p{latin}')

def num_script_chars():
    return pl.col('text').str.count_matches(r'\p{L}')

def num_latin_spans():
    return pl.col('text').str.count_matches(r'\p{latin}+')

def num_script_spans():
    return pl.col('text').str.count_matches(r'\p{L}+')

def words():
    return pl.col('text').str.extract_all(r'\S+')

def num_words():
    return words().list.len()

def latin_word_ratio():
    ws = words()
    return ratio(ws.list.eval(pl.element().str.contains('\p{latin}')).list.sum(), ws.list.len())

def latin_script_ratio():
    return ratio(num_latin_chars(), num_script_chars())

def average_word_len():
    return words().list.eval(pl.element().str.len_chars()).list.mean().cast(pld.Float32)

def hashbang_ratio():
    return ratio(pl.col('text').str.count_matches('#', literal=True), num_words())

def ellipsis_ratio():
    return ratio(pl.col('text').str.count_matches(r'\u2026|\.\.\.|\. \. \.'), num_words())

def non_empty_lines():
    return pl.col('text').str.extract_all(r'(?m:^.*\S.*$)')

def num_non_empty_lines():
    return non_empty_lines().list.len()

def init_bullet_ratio():
    return ratio(pl.col('text').str.count_matches(r'(?m:^\s*(\u25cf|\u2022|\u002a|\u002d))'), num_non_empty_lines())

def short_line_ratio():
    return non_empty_lines().list.eval(pl.element().str.strip_chars().str.len_chars() < 30).list.mean().cast(pld.Float32)

def words_per_line():
    return ratio(num_words(), num_non_empty_lines())

def end_ellipsis_ratio():
    return ratio(pl.col('text').str.count_matches(r'(?m:(?:\u2025|\.\.\.|\. \. \.)\s*$)'), num_non_empty_lines())

mapping = locals()

PREFILTER_BOUNDS = {
        'latin_word_ratio': Bound(low=.8),
        'latin_script_ratio': Bound(low=.5),
        'num_words': Bound(low=50, high=100000),
        'average_word_len': Bound(low=3, high=15),
        'hashbang_ratio': Bound(high=0.1),
        'ellipsis_ratio': Bound(high=0.1),
        'init_bullet_ratio': Bound(high=0.9),
        'end_ellipsis_ratio': Bound(high=0.3),
        'short_line_ratio': Bound(high=0.67),
        'words_per_line':  Bound(low=10/3),
        }

df = pl.scan_parquet('out/txt/CC-MAIN-20241003094020-20241003124020-00004.warc.parquet')

def alias(*fs):
    return (f().alias(f.__name__) for f in fs)

def mk_filter(**kwargs):
    return (v(pl.col(k)) for k, v in kwargs.items())

def mk_filter(**kwargs):
    return functools.reduce(lambda a, b: a & b, (v(pl.col(k)) for k, v in kwargs.items()))

def apply_prefilter(inpath, outpath):
    filtered = pl.scan_parquet(inpath).with_columns(*alias(
        latin_word_ratio, latin_script_ratio, num_words, average_word_len, 
        hashbang_ratio, ellipsis_ratio, init_bullet_ratio, end_ellipsis_ratio,
        short_line_ratio, words_per_line,
        )).filter(mk_filter(**PREFILTER_BOUNDS)).drop(*PREFILTER_BOUNDS)
    filtered.sink_parquet(outpath, statistics=False)
