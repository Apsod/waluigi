import unicodedata

import polars as pl
import polars.datatypes as pld
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.html import HTMLTree

def maybe_update(dict, key, val):
    if val is not None:
        dict[key] = val

def get_html_stuff(html):
    tree = HTMLTree.parse(html)

    metas = {name: None for name in [
        'tdm-policy', 
        'tdm-reservation', 
        'keywords', 
        'description',
        ]}
    
    if tree.head is not None:
        for match in tree.head.query_selector_all(
                ','.join(f'meta[name={name}]' for name in list(metas))
                ):
            metas[match.getattr('name')] = match.getattr('content')
    
    text = unicodedata.normalize('NFC', extract_plain_text(html, main_content=True, preserve_formatting=True))
    metas['meta-tdm-policy'] = metas.pop('tdm-policy')
    metas['meta-tdm-reservation'] = metas.pop('tdm-reservation')

    return {'text': text, **metas}

def first(a, b):
    return pl.when(a.is_not_null()).then(a).otherwise(b)

def extract_text(infile, outfile):
    df = pl.scan_parquet(infile)

    mapped = df.select(pl.col('html').map_elements(
            get_html_stuff, 
            return_dtype=pld.Struct({
                'text': pld.String(),
                'meta-tdm-policy': pld.String(),
                'meta-tdm-reservation': pld.String(),
                'keywords': pld.String(),
                'description': pld.String(),
                }),
            strategy='threading',
            )).unnest('html')

        
    combined = pl.concat([df.drop('html'), mapped], how='horizontal')

    combined = combined.with_columns(
            first(pl.col('meta-tdm-reservation'), pl.col('tdm-reservation')).alias('tdm-reservation'),
            first(pl.col('meta-tdm-policy'), pl.col('tdm-policy')).alias('tdm-policy'),
            ).drop(['meta-tdm-reservation', 'meta-tdm-policy'])
    
    combined.collect(streaming=True).write_parquet(outfile, statistics=False)
