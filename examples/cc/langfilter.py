import polars as pl
import polars.datatypes as pld
import fasttext
import numpy

LANGSET = [
    'eng', # English
    'ang', # Old English
    'sco', # Scots
    'deu', # German
    'gsw', # Swiss German
    'swg', # Swabian
    'ksh', # Kölsch
    'nds', # Low German
    'bar', # Bavarian
    'stq', # Saterfriesisch
    'gos', # Gronings
    'hrx', # Hunsrik
    'pdc', # Pennsylvania German
    'pfl', # Pfaelzisch
    'prg', # Prussian
    'got', # Gothic
    'gmh', # Middle High German
    'nld', # Dutch
    'afr', # Afrikaans
    'fry', # Western Frisian
    'frr', # Northern Frisian
    'ltz', # Luxembourgish
    'vls', # Vlaams
    'lim', # Limburgish
    'zea', # Zeelandic
    'swe', # Swedish
    'dan', # Danish
    'nno', # Norwegian (Nynorsk)
    'nob', # Norwegian (Bokmål)
    'isl', # Icelandic
    'fao', # Faroese
    'non', # Old Norse
    ]

LABELSET = set([f'__label__{code}_Latn' for code in LANGSET])


class FT(object):
    def __init__(self, path, labels=LABELSET):
        self.model = fasttext.load_model(path)
        lset = set(labels)
        self.indices = []
        self.labels = []
        for i, l in enumerate(self.model.labels):
            if l in lset:
                self.indices.append(i)
                self.labels.append(l)

    def predict(self, txt):
        scores = self.model.get_output_matrix() @ self.model.get_sentence_vector(txt)
        scores = numpy.exp(scores[self.indices] - numpy.logaddexp.reduce(scores))
        return {
                'top_label': self.labels[scores.argmax().item()],
                'top_score': scores.max().item(),
                'total_score': scores.sum().item(),
                }

    def predict_batch(self, txts):
        scores = self.model.get_output_matrix() @ numpy.stack([self.model.get_sentence_vector(txt) for txt in txts], axis=1)
        scores = numpy.exp(scores[self.indices] - numpy.logaddexp.reduce(scores, axis=0, keepdims=True))
        return pl.DataFrame({
                'top_label': [self.labels[i] for i in scores.argmax(0).tolist()],
                'top_score': scores.max(0),
                'total_score': scores.sum(0),
                })


def apply_langid(infile, outfile, model, top_lb=0.3, tot_lb=0.8):
    df = pl.scan_parquet(infile)
    filtered = df.with_columns(pl.col('text').str.replace_all(r'\s+', r' ').map_elements(
        model.predict,
        return_dtype=pld.Struct({
            'top_label': pld.String(),
            'top_score': pld.Float64(),
            'total_score': pld.Float64(),
            }),
        strategy='thread_local').struct.unnest()).filter(
                (pl.col('top_score') > top_lb) | (pl.col('total_score') > tot_lb)
        )
    filtered.sink_parquet(outfile, statistics=False)
