from typing import TYPE_CHECKING, Callable, Literal
import pyarrow.parquet as pq
import dask
import pyarrow as pa
from fastwarc.warc import ArchiveIterator, WarcRecordType, WarcRecord
from fastwarc.stream_io import *
from concurrent.futures import ProcessPoolExecutor

import multiprocessing as mp



def warc_iterator(infile, min_length=512):
    stream = GZipStream(FileStream(str(infile), 'rb'))
    iterator = ArchiveIterator(stream, 
                               record_types=WarcRecordType.response,
                               min_content_length=min_length,
                               )
    for record in iterator:
        record.freeze()
        yield record

def process_and_write(infile, outfile, min_length=512):
    schema = pa.schema({
        'html': pa.string(),
        'source': pa.string(),
        'record_id': pa.string(),
        'offset': pa.uint64(),
        'original_charset': pa.string(),
        'url': pa.string(),
        'date': pa.string(),
        'tdm-reservation': pa.string(),
        'tdm-policy': pa.string(),
        })

        
    group_size = 1024*1
    
    source = str(infile.absolute())

    with pq.ParquetWriter(
            outfile, 
            schema,
            sorting_columns=[pq.SortingColumn(3)],
            write_statistics=False,
            ) as writer:
        size = 0 
        batch = {k: [] for k in schema.names}
        
        for record in warc_iterator(infile, min_length):
            row = to_row(record)
            if row is not None:
                size += 1
                for k, v in row.items():
                    batch[k].append(v)
                batch['source'].append(source)
                if size == group_size:
                    writer.write_batch(pa.record_batch(batch, schema=schema))
                    size = 0
                    batch = {k: [] for k in schema.names}
        if size > 0: # write last batch if the batch has any leftover rows
            writer.write_batch(pa.record_batch(batch, schema=schema))

def to_row(record : "WarcRecord") -> dict | None:
    """Process a WARC record to extract the html and metadata (id, url, date)."""
    import cchardet
    import magic

    # content type filtering
    mime_type = record.headers.get("WARC-Identified-Payload-Type", None)
    if mime_type is not None and mime_type != "text/html":
        return

    record.parse_http()
    content_bytes = record.reader.read()
    if mime_type is None:
        # fallback for older crawls without payload types
        mime_type = magic.from_buffer(content_bytes, mime=True)
        if mime_type != "text/html":
            return

    # Decode the response bytes
    charset = "UTF-8"
    try:
        html = content_bytes.decode(charset)
    except UnicodeDecodeError:
        encoding_det = cchardet.detect(content_bytes)["encoding"]
        if not encoding_det or encoding_det == charset:
            return
        charset = encoding_det

        try:
            html = content_bytes.decode(charset)
        except (UnicodeDecodeError, LookupError):
            return

    id_ = record.headers["WARC-Record-ID"]
    url = record.headers.get("WARC-Target-URI", None)
    date = record.headers.get("WARC-Date", None)
    offset = record.stream_pos
    charset = charset
    tdm_reservation = record.http_headers.get('tdm-reservation', None)
    tdm_policy = record.http_headers.get('tdm-policy', None)

    # handle older formats
    if not url:
        url = record.headers["uri"]
    if not date:
        date = record.headers["archive-date"]

    return {"html": html, "record_id": id_, "offset": offset, "original_charset": charset, "url": url, "date": date, "tdm-reservation": tdm_reservation, "tdm-policy": tdm_policy}
