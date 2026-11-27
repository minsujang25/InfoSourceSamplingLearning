"""Validate regenerated provenance indexes, not immutable source bytes.
References must reproduce the original archived digests. Only explicitly named
floating fields may differ at the existing numerical tolerance; every other
field, missing cell, header, row count and row order must match exactly.
"""
import csv
import hashlib
import math
from pathlib import Path

FLOAT_FIELDS = {
    'v61_exhibit_source_rows.csv': {'mean', 'lo', 'hi'},
    'v62_exhibit_source_rows.csv': {'mean', 'lo', 'hi', 'mcse', 'negative', 'positive', 'zero', 'pooled_denominator', 'pooled_hi', 'pooled_lo', 'pooled_mean', 'estimate', 'category_share', 'conditional_mean', 'median', 'q25', 'q75'},
    'v63_exhibit_source_rows.csv': {'mean', 'lo', 'hi', 'mcse', 'positive', 'negative', 'zero', 'p_low', 'p_high', 'c_low', 'c_high'},
}
ATOL = RTOL = 1e-10


def compare_index(got, expected, archived_sha256):
    got, expected = Path(got), Path(expected)
    floats = FLOAT_FIELDS[expected.name]
    if hashlib.sha256(expected.read_bytes()).hexdigest() != archived_sha256:
        raise ValueError(f'Archived reference hash mismatch: {expected}')
    with got.open(newline='', encoding='utf-8') as a, expected.open(newline='', encoding='utf-8') as b:
        left, right = list(csv.reader(a)), list(csv.reader(b))
    if not left or not right or left[0] != right[0] or len(left) != len(right):
        raise ValueError(f'Index schema or row count mismatch: {got}')
    header = right[0]
    if len(set(header)) != len(header) or not floats <= set(header):
        raise ValueError(f'Unexpected index columns: {got}')
    maxerr = 0.0
    changed = 0
    for i, (x, y) in enumerate(zip(left[1:], right[1:]), 1):
        if len(x) != len(header) or len(y) != len(header):
            raise ValueError(f'{got}: malformed row {i}')
        for col, observed, reference in zip(header, x, y):
            if col in floats and observed != '' and reference != '':
                u, v = float(observed), float(reference)
                err = abs(u-v)
                ok = math.isfinite(u) and math.isfinite(v) and err <= ATOL + RTOL*abs(v)
                if ok:
                    maxerr = max(maxerr, err)
                    changed += observed != reference
            else:
                ok = observed == reference
            if not ok:
                raise ValueError(f'{got}: data row {i}, column {col}: observed={observed!r}, reference={reference!r}')
    return dict(rows=len(right)-1, changed_float_fields=changed, max_abs_residual=maxerr)
