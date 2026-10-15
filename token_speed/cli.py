import argparse
from contextlib import closing
import time
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from . import __version__
from .report import SCHEMA_VERSION, PREFILL_LENGTHS, fmt, save, measurement, render_console
from .runner import inspect, sample, schedule, run_samples, same_environment
from .scenarios import SEED, suite_hash, workload, make_prompt, tokenizer, estimate_usage


DEFAULT_MODELS = ("gpt-6.1-sol", "gpt-6-astra", "gpt-6-luna")

def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def positive_seconds(value):
    number = float(value)
    if not 0 < number < float('inf'):
        raise argparse.ArgumentTypeError('must be finite and positive')
    return number


def parser():
    p = argparse.ArgumentParser(allow_abbrev=False, description="Benchmark Codex prefill and decode.")
    p.add_argument("--version", action="version", version=__version__)
    commands = p.add_subparsers(dest="action", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--codex", default=os.environ.get("CODEX_BIN", "codex"), help="Codex executable path")
    common.add_argument("--profile", help="Codex config.toml profile")
    common.add_argument("--config", action="append", default=[], help="Codex key=value override")
    commands.add_parser("doctor", parents=[common], allow_abbrev=False, help="Inspect settings and model catalog")
    run = commands.add_parser("run", parents=[common], allow_abbrev=False, help="Run a single-length benchmark")
    run.add_argument("--model", action="append", help="Model ID; repeat to select multiple models")
    run.add_argument("--effort", default="low", help="Reasoning effort (default: low)")
    run.add_argument("--fast", choices=["inherit", "on", "off", "both"], default="both")
    run.add_argument("--input-tokens", type=nonnegative, default=32768, help="Input reference tokens (default: 32768)")
    run.add_argument("--output-tokens", type=nonnegative, default=512, help="Decode reference token target (default: 512)")
    run.add_argument("--timeout", type=positive_seconds, default=30, help="Per-request inference timeout in seconds (default: 30)")
    run.add_argument("--repeats", type=nonnegative, help="Samples per model/mode (default: 5; 2 for input >= 65536)")
    run.add_argument("--concurrency", type=nonnegative, default=5, help="Concurrent cases per model/mode (default: 5; 1 for serial)")
    run.add_argument("--yes", action="store_true", help="Skip confirmation after displaying estimated token usage")
    run.add_argument("--json", type=Path, help="Optional new JSON output file")
    return p


def select_models(args, info):
    catalog = {m['model'] for m in info['models']}
    candidates = list(dict.fromkeys(args.model or DEFAULT_MODELS))
    selected = [m for m in candidates if args.model or m in catalog]
    skipped = [{'model': m, 'reason': 'not listed by current Codex model/list'}
               for m in candidates if m not in selected]
    return selected, skipped


CASE_HEADERS = ('Case', 'Model', 'Fast', 'Passed', 'Input (tok)', 'Cached (tok)', 'Prefill (s)', 'Output (tok)', 'Decode (tps)', 'Reason')


def case_widths(models, count, fast_modes):
    return (max(len('Case'), len(str(count))),
            max([len('Model'), *(len(m) for m in models)]),
            max([len('Fast'), *(len(mode) for mode in fast_modes)]),
            6, 11, 12, 11, 12, 12)


def case_cells(values, widths):
    return ' | '.join(str(v).ljust(w) for v, w in zip(values[:-1], widths)) + ' | ' + values[-1]


def case_header(widths):
    return case_cells(CASE_HEADERS, widths) + '\n' + '-+-'.join('-' * w for w in (*widths, len('Reason')))


def case_line(index, row, widths):
    failure = ' '.join((row['failure'] or '-').split())
    usage = next((turn['usage'] for turn in row['turns'] if turn['stage'] == 'prefill'), None)
    counts = [usage[field] if usage is not None else None for field in PREFILL_LENGTHS.values()]
    output = measurement(row, 'output_tokens')
    return case_cells([str(index), row['model'], row['fast'].upper(),
                       'yes' if row['valid'] else 'no',
                       *('N/A' if count is None else str(count) for count in counts),
                       fmt(measurement(row, 'ttft_s')), 'N/A' if output is None else str(output),
                       fmt(measurement(row, 'decode_tok_s')),
                       failure], widths)


def main(argv):
    p = parser()
    args = p.parse_args(argv)
    binary = shutil.which(args.codex)
    if not binary:
        p.error('Codex not found; set --codex or CODEX_BIN.')
    args.codex = str(Path(binary).resolve())
    if args.action == 'doctor':
        print(json.dumps(inspect(args), ensure_ascii=False, indent=2))
        return 0
    if args.input_tokens < 256:
        p.error('--input-tokens must be >= 256')
    if args.output_tokens < 64:
        p.error('--output-tokens must be >= 64')
    if args.repeats is None:
        args.repeats = 2 if args.input_tokens >= 65536 else 5
    if args.repeats < 1:
        p.error('--repeats must be >= 1')
    if args.concurrency < 1:
        p.error('--concurrency must be >= 1')
    tokenizer()
    make_prompt(args.input_tokens, SEED, '0'*32)
    if args.json:
        args.json = args.json.resolve()
        if args.json.exists():
            p.error('--json file already exists')
    version = subprocess.run([args.codex, '--version'], capture_output=True, text=True, timeout=15, check=True).stdout.strip()
    info = inspect(args)
    models, skipped = select_models(args, info)
    plan = list(schedule(models, args.fast, args.repeats))
    estimate = estimate_usage(args, len(plan))
    print(f"ESTIMATE requests={estimate['requests']} input_tokens~{estimate['input_tokens']:,} "
          f"cached_input_tokens~{estimate['cached_input_tokens']:,} "
          f"output_tokens~{estimate['output_tokens']:,} total_tokens~{estimate['total_tokens']:,} "
          f"base_input_tokens_per_request~{estimate['base_input_tokens_per_request']:,} reasoning_tokens=unknown", flush=True)
    if plan and not args.yes:
        try:
            confirmed = input('Continue? [y/N]: ').strip().lower() in ('y', 'yes')
        except (EOFError, KeyboardInterrupt):
            confirmed = False
        if not confirmed:
            print('CANCELLED', flush=True)
            return 1
    data = {'schema_version': SCHEMA_VERSION, 'tool_version': __version__, 'created_at': datetime.now(timezone.utc).isoformat(),
            'suite_hash': suite_hash(), 'codex_version': version,
            'requested': {'models': args.model or list(DEFAULT_MODELS), 'effort': args.effort, 'fast': args.fast},
            'selected_models': models, 'skipped_models': skipped, 'workload': workload(args),
            'repeats': args.repeats, 'samples': [], 'status': 'running',
            'planned_samples': len(plan), 'server_tier_verified': False, 'estimated_usage': estimate}
    print(f"RUN effort={args.effort} input_tokens={args.input_tokens} output_tokens={args.output_tokens} repeats={args.repeats} cases={len(plan)} concurrency={args.concurrency} timeout={args.timeout:g}s", flush=True)
    for skip in skipped:
        print(f"SKIP model={skip['model']}: {skip['reason']}", flush=True)
    widths = case_widths(models, len(plan),
                         ['off', 'on'] if args.fast == 'both' else [args.fast])
    if plan:
        print(case_header(widths), flush=True)
    save(data, args.json)
    run_start = time.perf_counter()
    try:
        with closing(run_samples(args, plan, run_case=sample)) as results:
            for i, row in results:
                reference = next((r for r in data['samples'] if r['model'] == row['model'] and r['valid']), None)
                if row['valid'] and reference and not same_environment(reference, row, check_tier=args.fast != 'both'):
                    row['valid'] = False
                    row['failure'] = 'configuration/environment changed during run'
                data['samples'].append(row)
                save(data, args.json)
                print(case_line(i, row, widths), flush=True)
        data['status'] = 'completed' if models else 'no_available_models'
    except KeyboardInterrupt:
        data['status'] = 'interrupted'
    finally:
        data['elapsed_s'] = time.perf_counter() - run_start
        save(data, args.json)
        print(render_console(data), flush=True)
    if args.json:
        print(f'JSON: {args.json}')
    return 0 if data['status'] == 'completed' and data['samples'] and all(r['valid'] for r in data['samples']) else 1


def entrypoint():
    raise SystemExit(main(sys.argv[1:]))
