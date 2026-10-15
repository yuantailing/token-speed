import json
import math
import statistics

from .metrics import TOKEN_KEYS

SCHEMA_VERSION = 10

# Two-sided Student-t 95% critical values, df=1..30.
T95 = (12.706205, 4.302653, 3.182446, 2.776445, 2.570582, 2.446912,
       2.364624, 2.306004, 2.262157, 2.228139, 2.200985, 2.178813,
       2.160369, 2.144787, 2.131450, 2.119905, 2.109816, 2.100922,
       2.093024, 2.085963, 2.079614, 2.073873, 2.068658, 2.063899,
       2.059539, 2.055529, 2.051831, 2.048407, 2.045230, 2.042272)


def distribution(values):
    n = len(values)
    mean = statistics.mean(values) if n else None
    half = None
    if n >= 2:
        df = n - 1
        z = 1.959963984540054
        # Large-df expansion of the t quantile; accurate to displayed precision.
        critical = T95[df-1] if df <= 30 else (z + (z**3+z)/(4*df)
                    + (5*z**5+16*z**3+3*z)/(96*df**2)
                    + (3*z**7+19*z**5+17*z**3-15*z)/(384*df**3))
        half = critical * statistics.stdev(values) / math.sqrt(n)
    return {'n': n, 'mean': mean, 'ci95_half_width': half,
            'ci95': [mean-half, mean+half] if half is not None else None}


METRICS = {'ttft_s': 'prefill', 'decode_tok_s': 'decode', 'output_tokens': 'decode'}
PREFILL_LENGTHS = {'prefill_input_tokens': 'inputTokens', 'prefill_cached_tokens': 'cachedInputTokens'}


def measurement(row, key):
    return next((turn[key] for turn in row['turns'] if turn['stage'] == METRICS[key]), None)


def summarize(rows):
    groups = []
    for model, fast in dict.fromkeys((r['model'], r['fast']) for r in rows):
        cases = [r for r in rows if (r['model'], r['fast']) == (model, fast)]
        prefill_usage = [next(t for t in r['turns'] if t['stage'] == 'prefill')['usage']
                         for r in cases if r['valid']]
        groups.append({'model': model, 'fast': fast, 'attempted': len(cases),
                       'effort': cases[0]['effort'],
                       'passed': sum(r['valid'] for r in cases),
                       **{key: distribution([usage[field] for usage in prefill_usage])
                          for key, field in PREFILL_LENGTHS.items()},
                       **{k: distribution([measurement(r, k) for r in cases if r['valid']]) for k in METRICS}})
    turns = [turn for row in rows for turn in row['turns']]
    usages = [turn['usage'] for turn in turns if turn['usage'] is not None]
    totals = {key: sum(usage[key] for usage in usages) if usages or not turns else None
              for key in TOKEN_KEYS}
    return {'groups': groups, 'requests_sent': len(turns),
            'reported_usage_totals': totals, 'requests_missing_usage': len(turns) - len(usages)}


def fmt(value):
    return 'N/A' if value is None else f'{value:.2f}'


def interval(stats, precision):
    if stats['mean'] is None:
        return 'N/A'
    half = stats['ci95_half_width']
    return f"{stats['mean']:.{precision}f} ± {half:.{precision}f}" if half is not None else f"{stats['mean']:.{precision}f} ± N/A"


def table(headers, rows):
    rows = [[str(v) for v in row] for row in [headers, *rows]]
    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    lines = [' | '.join(v.ljust(w) for v, w in zip(row, widths)) for row in rows]
    lines.insert(1, '-+-'.join('-'*w for w in widths))
    return '\n'.join(lines)


def usage_line(data, summary):
    totals = summary['reported_usage_totals']
    fields = [('input_tokens', 'inputTokens'), ('cached_input_tokens', 'cachedInputTokens'),
              ('output_tokens', 'outputTokens'), ('reasoning_output_tokens', 'reasoningOutputTokens'),
              ('total_tokens', 'totalTokens')]
    values = [f'{label}={totals[key]:,}' if totals[key] is not None else f'{label}=N/A'
              for label, key in fields]
    complete = (data['status'] in ('completed', 'no_available_models')
                and all(turn['status'] == 'completed' for row in data['samples'] for turn in row['turns'])
                and summary['requests_missing_usage'] == 0)
    return ('USAGE ' + ' '.join(values) + f" complete={'yes' if complete else 'no'}"
            + f" requests_missing_usage={summary['requests_missing_usage']}")


def render_console(data):
    summary = summarize(data['samples'])
    return '\n'.join([
        f"\nSUMMARY status={data['status']} (mean ± 95% CI)",
        table(['Model', 'Fast', 'Passed', 'Input (tok)', 'Cached (tok)', 'Prefill (s)', 'Output (tok)', 'Decode (tps)'], [
            [g['model'], g['fast'].upper(), f"{g['passed']}/{g['attempted']}",
             interval(g['prefill_input_tokens'], 0), interval(g['prefill_cached_tokens'], 0),
             interval(g['ttft_s'], 2), interval(g['output_tokens'], 0), interval(g['decode_tok_s'], 2)] for g in summary['groups']]),
        usage_line(data, summary)])


def save(data, output):
    data['summary'] = summarize(data['samples'])
    if output is not None:
        temporary = output.with_name(output.name+'.tmp')
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
        temporary.replace(output)
