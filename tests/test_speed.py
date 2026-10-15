import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from token_speed.cli import parser, main, select_models, DEFAULT_MODELS, case_widths, case_header, case_line
from token_speed.metrics import TOKEN_KEYS, TurnMetrics, run_turn
from token_speed.report import SCHEMA_VERSION, summarize, render_console, distribution
from token_speed.rpc import AppServer, ProtocolError
from token_speed.runner import sample, schedule, verify_settings, read_environment, command, cache_evidence, run_samples
from token_speed.scenarios import make_prompt, tokenizer, validate, validate_ack, decode_prompt, decode_text, estimate_usage

FAKE = Path(__file__).with_name('fake_codex.py')


def emit(metrics, at, method, **params):
    metrics.event(at, {'method': method, 'params': {'threadId': 't', 'turnId': 'u', **params}})


class InputTest(unittest.TestCase):
    def test_input_lengths_are_exact_and_deterministic(self):
        for n in (256, 8192, 32768, 65536):
            a = make_prompt(n, 42, 'abc'*10)
            self.assertEqual(len(tokenizer().encode(a)), n)
            self.assertEqual(a, make_prompt(n, 42, 'abc'*10))
            self.assertNotEqual(a, make_prompt(n, 42, 'def'*10))


class DecodeInputTest(unittest.TestCase):
    def test_progress_output_has_finite_end_and_short_followup(self):
        for target in (64, 512, 1024):
            expected = decode_text(target)
            self.assertGreaterEqual(len(tokenizer().encode(expected)), target + 16)
            self.assertLess(len(tokenizer().encode(decode_prompt(target))), 256)
            lines = expected.splitlines()
            count = len(lines) - 1
            self.assertEqual(lines[-1], 'END')
            for i, line in enumerate(lines[:-1], 1):
                state = 'DONE' if i == count else 'RUN'
                self.assertTrue(line.startswith(f'{i:02d}/{count:02d} {state}: '))
            self.assertIn(' '.join(f'{i:02d}' for i in range(1, count+1)), decode_prompt(target))
        self.assertEqual(len(tokenizer().encode(decode_text(512))), 529)

    def test_completion_requires_correct_final_state_and_stop_marker(self):
        expected = decode_text(512)
        prefix = tokenizer().decode(tokenizer().encode(expected)[:512])
        turn = {'status': 'completed', 'tool_calls': 0, 'rerouted': False,
                'message_count': 1, 'output_reference_tokens': 512,
                'answer': prefix, 'full_answer': expected, 'decode_tok_s': 100}
        self.assertTrue(validate(turn, 512)[0])
        for text in (expected.removesuffix('\nEND'), expected + '\n01/24 RUN: extra',
                     expected.replace('24/24 DONE', '24/24 RUN'),
                     expected.replace('02/24 RUN', '01/24 RUN')):
            with self.subTest(text=text[-50:]):
                self.assertFalse(validate({**turn, 'full_answer': text}, 512)[0])
        self.assertTrue(validate({**turn, 'full_answer': expected + '\n'}, 512)[0])


class MetricsTest(unittest.TestCase):
    def test_native_output_rate_uses_first_text_to_final_completion(self):
        baseline = {**dict.fromkeys(TOKEN_KEYS, 0), 'outputTokens': 10}
        m = TurnMetrics('t', 'u', 10, 4, baseline)
        emit(m, 12, 'item/agentMessage/delta', itemId='a', delta='alpha beta')
        emit(m, 14, 'item/agentMessage/delta', itemId='a', delta=' gamma delta')
        emit(m, 15, 'item/agentMessage/delta', itemId='a', delta=' tail')
        emit(m, 18, 'thread/tokenUsage/updated', tokenUsage={'total': {
            **dict.fromkeys(TOKEN_KEYS, 0), 'outputTokens': 28}})
        emit(m, 19, 'turn/completed', turn={'id': 'u', 'status': 'completed'})
        r = m.result()
        self.assertEqual(r['ttft_s'], 2)
        self.assertEqual(r['output_tokens'], 18)
        self.assertEqual(r['decode_s'], 7)
        self.assertEqual(r['decode_tok_s'], 18 / 7)
        self.assertEqual(r['elapsed_s'], 9)
        self.assertEqual(r['answer'], 'alpha beta gamma delta')
        self.assertEqual(r['full_answer'], 'alpha beta gamma delta tail')

    def test_native_rate_does_not_depend_on_chunk_boundaries(self):
        for chunks in (['alpha beta'], ['al', 'pha beta']):
            m = TurnMetrics('t', 'u', 0, 1, dict.fromkeys(TOKEN_KEYS, 0))
            for stamp, text in enumerate(chunks, 1):
                emit(m, stamp, 'item/agentMessage/delta', itemId='a', delta=text)
            emit(m, 3, 'thread/tokenUsage/updated', tokenUsage={'total': {
                **dict.fromkeys(TOKEN_KEYS, 0), 'outputTokens': 8}})
            emit(m, 4, 'turn/completed', turn={'id': 'u', 'status': 'completed'})
            self.assertEqual(m.result()['decode_tok_s'], 8 / 3)

    def test_native_rate_requires_usage_and_successful_completion(self):
        for status in ('completed', 'interrupted', 'failed'):
            m = TurnMetrics('t', 'u', 0, 1, dict.fromkeys(TOKEN_KEYS, 0))
            emit(m, 1, 'item/agentMessage/delta', itemId='a', delta='alpha')
            emit(m, 2, 'turn/completed', turn={'id': 'u', 'status': status})
            self.assertIsNone(m.result()['output_tokens'])
            self.assertIsNone(m.result()['decode_tok_s'])
            emit(m, 3, 'thread/tokenUsage/updated', tokenUsage={'total': {
                **dict.fromkeys(TOKEN_KEYS, 0), 'outputTokens': 4}})
            self.assertEqual(m.result()['decode_tok_s'], 4 if status == 'completed' else None)

    def test_native_rate_requires_first_text_and_positive_decode_duration(self):
        for first in (None, 2):
            m = TurnMetrics('t', 'u', 0, 1, dict.fromkeys(TOKEN_KEYS, 0))
            if first is not None:
                emit(m, first, 'item/agentMessage/delta', itemId='a', delta='alpha')
            emit(m, 2, 'thread/tokenUsage/updated', tokenUsage={'total': {
                **dict.fromkeys(TOKEN_KEYS, 0), 'outputTokens': 4}})
            emit(m, 2, 'turn/completed', turn={'id': 'u', 'status': 'completed'})
            self.assertIsNone(m.result()['decode_tok_s'])

    def test_missing_usage_is_not_zero(self):
        m = TurnMetrics('t', 'u', 0, 64, dict.fromkeys(TOKEN_KEYS, 0))
        self.assertIsNone(m.result()['usage'])
        with self.assertRaises(KeyError):
            emit(m, 1, 'thread/tokenUsage/updated', tokenUsage={'total': {'inputTokens': 100}})

    def test_other_threads_and_turns_are_ignored(self):
        m = TurnMetrics('t', 'u', 0, 64, dict.fromkeys(TOKEN_KEYS, 0))
        emit(m, 1, 'item/agentMessage/delta', threadId='other', itemId='a', delta='wrong')
        emit(m, 1, 'item/agentMessage/delta', turnId='previous', itemId='a', delta='wrong')
        self.assertEqual(m.text, '')

    def test_three_turn_usage_is_a_delta_not_thread_total(self):
        baseline = {**dict.fromkeys(TOKEN_KEYS, 0), 'inputTokens': 15000, 'cachedInputTokens': 12000, 'outputTokens': 5, 'reasoningOutputTokens': 0}
        m = TurnMetrics('t', 'u', 0, 1, baseline)
        emit(m, 1, 'thread/tokenUsage/updated', tokenUsage={'total': {**dict.fromkeys(TOKEN_KEYS, 0),
            'inputTokens': 38000, 'cachedInputTokens': 27000, 'outputTokens': 10, 'reasoningOutputTokens': 0}})
        self.assertEqual(m.result()['usage']['inputTokens'], 23000)
        self.assertEqual(m.result()['usage']['cachedInputTokens'], 15000)
        self.assertEqual(m.result()['usage']['outputTokens'], 5)
        self.assertEqual(m.result()['cumulative_usage']['inputTokens'], 38000)

    def test_ack_is_reference_text_not_a_fabricated_native_limit(self):
        m = TurnMetrics('t', 'u', 0, 1, dict.fromkeys(TOKEN_KEYS, 0))
        emit(m, 1, 'item/agentMessage/delta', itemId='a', delta='K')
        emit(m, 2, 'thread/tokenUsage/updated', tokenUsage={'total': {**dict.fromkeys(TOKEN_KEYS, 0), 'inputTokens': 15000, 'outputTokens': 5, 'reasoningOutputTokens': 0}})
        emit(m, 3, 'turn/completed', turn={'id': 'u', 'status': 'completed'})
        self.assertTrue(validate_ack(m.result())[0])
        self.assertEqual(m.result()['usage']['outputTokens'], 5)
        emit(m, 4, 'item/agentMessage/delta', itemId='a', delta='extra')
        self.assertFalse(validate_ack(m.result())[0])


class CacheTest(unittest.TestCase):
    def test_prefill_only_requires_positive_cache(self):
        prior = {'usage': {'inputTokens': 14421, 'cachedInputTokens': 14080}}
        for cached in (1, 11008, 14080, 22624):
            with self.subTest(cached=cached):
                current = {'usage': {'inputTokens': 22624, 'cachedInputTokens': cached}}
                self.assertTrue(cache_evidence(prior, current, False)['passed'])
        for cached in (None, 0, -1, True, '14080'):
            self.assertFalse(cache_evidence(prior, {'usage': {'cachedInputTokens': cached}}, False)['passed'])
        self.assertFalse(cache_evidence(prior, {'usage': None, 'status': 'completed', 'stage': 'warmup'}, False)['passed'])

    def test_decode_requires_nondecreasing_cache_not_input_coverage(self):
        prior = {'usage': {'inputTokens': 22624, 'cachedInputTokens': 14080}}
        for cached, expected in ((22272, True), (14081, True), (14080, True),
                                 (13056, False), (0, False), (None, False)):
            with self.subTest(cached=cached):
                current = {'usage': {'cachedInputTokens': cached}}
                self.assertEqual(cache_evidence(prior, current, True)['passed'], expected)
        for prior_cached in (None, -1, True):
            self.assertFalse(cache_evidence({'usage': {'cachedInputTokens': prior_cached}},
                                           {'usage': {'cachedInputTokens': 22272}}, True)['passed'])
        self.assertTrue(cache_evidence({'usage': {'cachedInputTokens': 0}},
                                       {'usage': {'cachedInputTokens': 22272}}, True)['passed'])

    def test_multi_model_pair_schedule(self):
        rows = list(schedule(DEFAULT_MODELS, 'both', 5))
        self.assertEqual(len(rows), 30)
        for model in DEFAULT_MODELS:
            selected = [r for r in rows if r[0] == model]
            self.assertEqual(sum(r[1] == 'on' for r in selected), 5)
            self.assertEqual(sum(r[1] == 'off' for r in selected), 5)
            self.assertEqual({r[2] for r in selected}, set(range(5)))


class ProtocolTest(unittest.TestCase):
    def test_pre_ack_delta_and_fixed_output_stop(self):
        with tempfile.TemporaryDirectory() as cwd, AppServer([sys.executable, str(FAKE)], cwd, 60, threading.Event()) as server:
            server.call('thread/start', {})
            row = run_turn(server, 'thread1', decode_prompt(64), 2, 64, dict.fromkeys(TOKEN_KEYS, 0), finish_timeout=2)
            self.assertTrue(row['target_reached'])
            self.assertGreater(row['decode_tok_s'], 0)
            self.assertLess(row['output_reference_tokens'], 200)
            self.assertTrue(validate(row, 64)[0])

    def test_finish_timeout_stops_a_stream_that_does_not_complete(self):
        with patch.dict('os.environ', {'FAKE_CODEX_MODE': 'hang-after-target'}):
            with tempfile.TemporaryDirectory() as cwd, AppServer([sys.executable, str(FAKE)], cwd, 60, threading.Event()) as server:
                server.call('thread/start', {})
                result = run_turn(server, 'thread1', 'test', 5, 64, dict.fromkeys(TOKEN_KEYS, 0), finish_timeout=.1)
                self.assertTrue(result['target_reached'])
                self.assertEqual(result['status'], 'timeout')
                self.assertEqual(result['timeout_phase'], 'completion')
                self.assertIn('completion timeout after', result['failure'])
                self.assertLess(result['elapsed_s'], 1)
                self.assertFalse(validate(result, 64)[0])

    def test_cancellation_interrupts_blocked_rpc_receive(self):
        cancel = threading.Event()
        with tempfile.TemporaryDirectory() as cwd, AppServer(
                [sys.executable, str(FAKE)], cwd, 60, cancel_event=cancel) as server:
            timer = threading.Timer(.05, cancel.set)
            timer.start()
            start = time.perf_counter()
            try:
                with self.assertRaises(InterruptedError):
                    server.receive(start + 30)
            finally:
                timer.join()
            self.assertLess(time.perf_counter() - start, 1)
        self.assertIsNotNone(server.proc.poll())

    def test_timeout_disconnect_and_interactive_request(self):
        for mode, status in [('timeout', 'timeout'), ('disconnect', 'transport_error'), ('interactive', 'transport_error')]:
            with self.subTest(mode=mode), patch.dict('os.environ', {'FAKE_CODEX_MODE': mode}):
                with tempfile.TemporaryDirectory() as cwd, AppServer([sys.executable, str(FAKE)], cwd, 60, threading.Event()) as server:
                    server.call('thread/start', {})
                    row = run_turn(server, 'thread1', 'test', .1, 64, dict.fromkeys(TOKEN_KEYS, 0), finish_timeout=2)
                    self.assertEqual(row['status'], status)
                self.assertIsNotNone(server.proc.poll())


class RunnerTest(unittest.TestCase):
    def args(self):
        args = parser().parse_args(['run', '--codex', str(FAKE), '--effort', 'high'])
        return args

    def test_output_target_and_timeout_are_applied_to_requests(self):
        args = self.args()
        args.output_tokens = 64
        args.timeout = 7.5
        with patch('token_speed.runner.run_turn', wraps=run_turn) as run:
            result = sample(args, 'chosen', 'off', 0, threading.Event())
        self.assertTrue(result['valid'])
        self.assertEqual([c.args[3] for c in run.call_args_list], [7.5, 7.5, 7.5])
        self.assertEqual([c.args[4] for c in run.call_args_list], [1, 1, 64])
        self.assertTrue(all(c.kwargs['finish_timeout'] == 2 for c in run.call_args_list))

    def test_prefill_cache_failure_survives_decode_completion_failure(self):
        args = self.args()
        with patch.dict('os.environ', {'FAKE_CODEX_MODE': 'cache-miss-and-hang'}):
            result = sample(args, 'chosen', 'off', 0, threading.Event())
        self.assertIn('prefill: cache check failed', result['failure'])
        self.assertIn('cached=0', result['failure'])
        self.assertIn('decode: completion timeout after', result['failure'])
        self.assertFalse(result['valid'])

    def test_decode_defaults_and_english_help(self):
        args = parser().parse_args(['run'])
        self.assertEqual(args.effort, 'low')
        self.assertEqual((args.input_tokens, args.output_tokens, args.timeout), (32768, 512, 30))
        with redirect_stdout(io.StringIO()) as stdout, self.assertRaises(SystemExit):
            parser().parse_args(['run', '--help'])
        self.assertNotRegex(stdout.getvalue(), r'[\u4e00-\u9fff]')

    def test_three_requests_in_each_context_scenario(self):
        for length in (8192, 65536):
            args = self.args()
            args.input_tokens = length
            row = sample(args, 'chosen', 'on', 0, threading.Event())
            self.assertTrue(row['valid'], row)
            self.assertEqual(len(row['turns']), 3)
            self.assertEqual([t['stage'] for t in row['turns']], ['warmup', 'prefill', 'decode'])
            self.assertTrue(row['cache_checks']['prefill']['passed'])
            self.assertTrue(row['cache_checks']['decode']['passed'])
            self.assertEqual(row['input_reference_tokens'], length)
            self.assertNotIn('answer', row['turns'][0])
            self.assertNotIn('DO_NOT_EXPORT', json.dumps(row))
            self.assertEqual(row['effective']['model'], 'chosen')

    def test_equal_cache_counts_are_accepted_end_to_end(self):
        with patch.dict('os.environ', {'FAKE_CODEX_MODE': 'decode-cache-equal'}):
            result = sample(self.args(), 'chosen', 'off', 0, threading.Event())
        self.assertTrue(result['valid'], result['failure'])

    def test_cache_boundary_gap_is_accepted_end_to_end(self):
        with patch.dict('os.environ', {'FAKE_CODEX_MODE': 'cache-boundary-gap'}):
            result = sample(self.args(), 'chosen', 'off', 0, threading.Event())
        self.assertTrue(result['valid'], result['failure'])

    def test_failed_and_nonstandard_samples_are_rejected(self):
        for mode in ('substitute-model', 'failed', 'reroute', 'short', 'tool', 'bad-ack', 'cache-miss', 'decode-cache-miss', 'decode-cache-decreased', 'no-usage'):
            with self.subTest(mode=mode), patch.dict('os.environ', {'FAKE_CODEX_MODE': mode}):
                row = sample(self.args(), 'chosen', 'on', 0, threading.Event())
                self.assertFalse(row['valid'])
                self.assertTrue(row['failure'])

    def test_missing_requested_setting_is_a_protocol_shape_error(self):
        with patch.dict('os.environ', {'FAKE_CODEX_MODE': 'ignore-tier'}):
            with self.assertRaises(KeyError):
                sample(self.args(), 'chosen', 'on', 0, threading.Event())

    def test_fast_off_and_pair_order(self):
        with self.assertRaises(ProtocolError):
            verify_settings({'model': 'chosen', 'reasoningEffort': 'high', 'serviceTier': 'fast'}, 'chosen', 'high', 'off')
        rows = list(schedule(['chosen', 'other'], 'both', 4))
        self.assertEqual(rows, [(model, mode, repeat) for model in ('chosen', 'other')
                                for mode in ('off', 'on') for repeat in range(4)])

    def test_cli_console_lines_and_optional_json(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as stdout:
            out = Path(tmp)/'run.json'
            code = main(['run', '--yes', '--codex', str(FAKE), '--model', 'chosen', '--model', 'other',
                         '--repeats', '2', '--json', str(out)])
            self.assertEqual(code, 0)
            data = json.loads(out.read_text())
            self.assertEqual(data['schema_version'], SCHEMA_VERSION)
            self.assertEqual(len(data['summary']['groups']), 4)
            self.assertEqual(len(data['samples']), 8)
            self.assertEqual(list(Path(tmp).iterdir()), [out])
            lines = stdout.getvalue().splitlines()
            self.assertEqual(sum(line.split('|', 1)[0].strip().isdigit() for line in lines), 8)
            self.assertIn('±', stdout.getvalue())
            self.assertNotIn('median', json.dumps(data))
            self.assertNotIn('HTML', stdout.getvalue())
            self.assertNotRegex(stdout.getvalue(), r'[\u4e00-\u9fff]')
            self.assertIn('Prefill (s)', stdout.getvalue())
            self.assertIn('Decode (tps)', stdout.getvalue())
            self.assertIn('Output (tok)', stdout.getvalue())
            self.assertEqual(data['summary']['groups'][0]['output_tokens']['mean'], 533)
            self.assertIn('cached_input_tokens~', stdout.getvalue())
            self.assertGreater(data['estimated_usage']['cached_input_tokens'], 0)
            self.assertNotIn('Repeat', stdout.getvalue())
            self.assertNotIn('Effort', stdout.getvalue())
            self.assertNotIn('(n=', stdout.getvalue())
            self.assertIn('RUN effort=low ', stdout.getvalue())
            self.assertNotIn('Fast 倍数', stdout.getvalue())

    def test_concurrency_is_bounded_and_model_mode_groups_do_not_overlap(self):
        args = self.args()
        self.assertEqual(args.concurrency, 5)
        barrier, lock = threading.Barrier(5), threading.Lock()
        active, peaks, seen = [], [], []
        def run(case_args, model, fast, repeat, cancel):
            group = (model, fast)
            with lock:
                active.append(group)
                self.assertTrue(all(g == group for g in active))
                peaks.append(len(active))
            barrier.wait(timeout=5)
            with lock:
                active.remove(group)
                seen.append((group, repeat))
            return {'model': model, 'fast': fast, 'repeat': repeat}
        plan = list(schedule(['a', 'b'], 'both', 5))
        results = list(run_samples(args, plan, run))
        self.assertEqual(len(results), 20)
        self.assertEqual(max(peaks), 5)
        self.assertEqual(sorted(i for i, _ in results), list(range(1, 21)))
        self.assertEqual([(r['model'], r['fast']) for _, r in results],
                         [(m, f) for m, f, _ in plan])

    def test_closing_parallel_run_cancels_active_and_queued_work(self):
        args = self.args()
        args.concurrency = 2
        started, stopped = threading.Event(), threading.Event()
        def run(case_args, model, fast, repeat, cancel):
            if repeat == 0:
                self.assertTrue(started.wait(5))
                return {'repeat': repeat}
            started.set()
            self.assertTrue(cancel.wait(5))
            stopped.set()
            return {'repeat': repeat}
        results = run_samples(args, list(schedule(['a'], 'off', 20)), run)
        self.assertEqual(next(results)[0], 1)
        results.close()
        self.assertTrue(stopped.is_set())

    def test_declined_or_missing_confirmation_never_starts_inference(self):
        self.assertFalse(parser().parse_args(['run']).yes)
        for reply in ('n', '', 'anything', EOFError(), KeyboardInterrupt()):
            with self.subTest(reply=reply), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / 'cancelled.json'
                options = {'side_effect': reply} if isinstance(reply, BaseException) else {'return_value': reply}
                with patch('builtins.input', **options) as confirm, patch('token_speed.cli.sample') as infer, redirect_stdout(io.StringIO()) as stdout:
                    code = main(['run', '--codex', str(FAKE), '--model', 'chosen', '--json', str(out)])
                self.assertEqual(code, 1)
                infer.assert_not_called()
                confirm.assert_called_once_with('Continue? [y/N]: ')
                self.assertIn('ESTIMATE', stdout.getvalue())
                self.assertIn('CANCELLED', stdout.getvalue())
                self.assertFalse(out.exists())

    def test_confirmation_and_yes_option(self):
        for flag in (None, '--yes'):
            with self.subTest(flag=flag), patch('builtins.input', return_value='YES') as confirm, redirect_stdout(io.StringIO()) as stdout:
                argv = ['run', '--codex', str(FAKE), '--model', 'chosen', '--fast', 'off',
                        '--repeats', '1']
                if flag:
                    argv.append(flag)
                self.assertEqual(main(argv), 0)
                self.assertEqual(confirm.call_count, 0 if flag else 1)
                self.assertLess(stdout.getvalue().index('ESTIMATE'), stdout.getvalue().index('RUN '))
                self.assertIn('USAGE input_tokens=', stdout.getvalue())
                self.assertIn('complete=yes', stdout.getvalue())

    def test_case_table_aligns_pass_fail_and_puts_reason_last(self):
        widths = case_widths(['short', 'a-long-relay-model-name'], 12, ['off', 'on'])
        a = row('off', 0, 1.25, 150, cached=100, valid=True)
        a['model'] = 'short'
        a['effort'] = 'high'
        b = row('on', 1, None, 120, valid=False, cached=100)
        b['model'] = 'a-long-relay-model-name'
        b['failure'] = 'decode: cache check failed\n  with additional details'
        lines = [case_header(widths).splitlines()[0], case_line(1, a, widths),
                 case_line(12, b, widths)]
        positions = [[i for i, c in enumerate(line) if c == '|'] for line in lines]
        self.assertEqual(positions[0], positions[1])
        self.assertEqual(positions[0], positions[2])
        self.assertEqual(widths[0], 4)
        self.assertEqual(widths[2], 4)
        self.assertEqual(lines[1].split('|')[0].strip(), '1')
        self.assertEqual(lines[2].split('|')[0].strip(), '12')
        self.assertEqual(case_widths(['a'], 1, ['inherit'])[2], 7)
        self.assertTrue(lines[0].endswith('Reason'))
        self.assertNotIn('Repeat', lines[0])
        self.assertTrue(lines[2].endswith('decode: cache check failed with additional details'))
        self.assertNotIn('[excluded]', '\n'.join(lines))
        self.assertEqual([c.strip() for c in lines[0].split('|')][3:9],
                         ['Passed', 'Input (tok)', 'Cached (tok)', 'Prefill (s)', 'Output (tok)', 'Decode (tps)'])
        self.assertEqual([c.strip() for c in lines[1].split('|')][3:9],
                         ['yes', '1000', '100', '1.25', '533', '150.00'])
        self.assertEqual([c.strip() for c in lines[2].split('|')][3:9],
                         ['no', '1000', '100', 'N/A', '533', '120.00'])

        b['turns'][0]['usage'] = None
        self.assertEqual([c.strip() for c in case_line(12, b, widths).split('|')][4:6], ['N/A', 'N/A'])
        b['turns'] = []
        self.assertEqual([c.strip() for c in case_line(12, b, widths).split('|')][4:9], ['N/A'] * 5)

    def test_default_models_filter_catalog_but_explicit_relays_are_allowed(self):
        args = parser().parse_args(['run'])
        self.assertEqual(args.fast, 'both')
        catalog = {'models': [{'model': DEFAULT_MODELS[1]}, {'model': 'unrelated'}]}
        models, skipped = select_models(args, catalog)
        self.assertEqual(models, [DEFAULT_MODELS[1]])
        self.assertEqual(len(skipped), 2)
        args.model = ['relay-model', 'relay-model']
        self.assertEqual(select_models(args, catalog), (['relay-model'], []))

    def test_no_available_defaults_exits_without_inference(self):
        with patch('token_speed.cli.sample') as infer, redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(main(['run', '--yes', '--codex', str(FAKE)]), 1)
            infer.assert_not_called()
            self.assertIn('no_available_models', stdout.getvalue())

    def test_length_controls_default_repeat_budget_and_no_default_artifacts(self):
        for length, repeats in [(8192, 5), (32768, 5), (65536, 2)]:
            with patch('token_speed.cli.sample', side_effect=KeyboardInterrupt), patch('token_speed.cli.save') as save, redirect_stdout(io.StringIO()):
                self.assertEqual(main(['run', '--yes', '--codex', str(FAKE), '--model', 'chosen', '--input-tokens', str(length)]), 1)
                data, output = save.call_args.args
                self.assertIsNone(output)
                self.assertEqual(data['planned_samples'], repeats*2)
                self.assertEqual(data['workload']['input_tokens'], length)

    def test_zero_concurrency_fails_before_inference(self):
        with patch('token_speed.cli.sample') as infer, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(['run', '--yes', '--codex', str(FAKE), '--concurrency', '0'])
            infer.assert_not_called()

    def test_invalid_output_target_and_timeout_fail_before_inference(self):
        options = [('--output-tokens', '63'), *[('--timeout', value) for value in ('0', '-1', 'nan', 'inf')]]
        for option, value in options:
            with self.subTest(option=option, value=value), patch('token_speed.cli.sample') as infer, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(['run', '--codex', str(FAKE), option, value])
                infer.assert_not_called()

    def test_only_canonical_options_are_accepted(self):
        removed = [('--length', '8192'), ('--seed', '42'),
                   ('--turn-delay', '0'), ('--decode-timeout', '30'),
                   ('--finish-timeout', '2'), ('--setup-timeout', '60'), ('--work-root', '/tmp'),
                   ('--require-native-limit',), ('-y',), ('-c', 'model="test"'), ('--input', '8192')]
        for options in removed:
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser().parse_args(['run', *options])


    def test_any_cache_failure_invalidates_whole_sample(self):
        for mode, expected in [('cache-miss', (False, True)), ('decode-cache-miss', (True, False))]:
            with self.subTest(mode=mode), patch.dict('os.environ', {'FAKE_CODEX_MODE': mode}):
                result = sample(self.args(), 'chosen', 'on', 0, threading.Event())
                self.assertEqual(len(result['turns']), 3)
                self.assertEqual(tuple(result['cache_checks'][stage]['passed']
                                       for stage in ('prefill', 'decode')), expected)
                group = summarize([result])['groups'][0]
                self.assertEqual(group['ttft_s']['n'], 0)
                self.assertEqual(group['decode_tok_s']['n'], 0)
                self.assertFalse(result['valid'])
    def test_existing_provider_and_proxy_are_preserved_without_exporting_keys(self):
        class ConfigServer:
            def call(self, method, params):
                if method == "config/read":
                    return {"config": {"model_provider": "relay", "model_providers": {"relay": {
                        "base_url": "https://user:secret@relay.example/v1?key=secret",
                        "env_key": "PROVIDER_KEY", "requires_openai_auth": False,
                        "http_headers": {"Authorization": "Bearer secret"}}}}}
                raise AssertionError(f"Unexpected RPC: {method}")
        with patch.dict("os.environ", {"PROVIDER_KEY": "secret", "HTTPS_PROXY": "http://secret:secret@proxy"}):
            info = read_environment(ConfigServer(), Path.cwd())
            self.assertEqual(info["provider"], "relay")
            self.assertNotIn("secret", json.dumps(info))
        args = self.args()
        args.profile = "relay"
        args.config = ['model_provider="relay"']
        cmd = command(args, "on")
        self.assertIn('profile="relay"', cmd)
        self.assertIn('model_provider="relay"', cmd)

    def test_fast_override_null_defaults_do_not_change_context_fingerprint(self):
        class ConfigServer:
            def __init__(self, features):
                self.features = features
            def call(self, method, params):
                assert method == "config/read"
                return {"config": {"features": self.features}}
        off = read_environment(ConfigServer({"tool_feature": True}), Path.cwd())
        on = read_environment(ConfigServer({"tool_feature": True, "fast_mode": True, "network_proxy": None}), Path.cwd())
        self.assertEqual(off["context_fingerprint"], on["context_fingerprint"])
        changed = read_environment(ConfigServer({"tool_feature": False}), Path.cwd())
        self.assertNotEqual(off["context_fingerprint"], changed["context_fingerprint"])


def row(fast, repeat, ttft, decode, *, valid, cached):
    return {'fast': fast, 'repeat': repeat, 'model': 'test', 'valid': valid, 'effort': 'low',
            'failure': None, 'input_reference_tokens': 8192,
            'effective': {'model': 'test', 'serviceTier': fast, 'modelProvider': 'openai',
                          'reasoningEffort': 'low', 'approvalPolicy': 'never', 'sandbox': 'readOnly'},
            'environment': dict.fromkeys(('route_fingerprint',
                                         'context_fingerprint', 'platform', 'python'), 'test'),
            'instruction_fingerprint': 'test', 'cache_checks': {'prefill': {'passed': valid}, 'decode': {'passed': valid}},
            'turns': [{'stage': 'prefill', 'ttft_s': ttft, 'status': 'completed',
                       'usage': {**dict.fromkeys(TOKEN_KEYS, 0), 'inputTokens': 1000, 'totalTokens': 1000, 'cachedInputTokens': cached, 'reasoningOutputTokens': 0}},
                      {'stage': 'decode', 'decode_tok_s': decode, 'output_tokens': 533, 'usage': None, 'status': 'completed'}]}


class UsageTest(unittest.TestCase):
    def test_estimate_counts_history_and_entire_expected_output(self):
        args = parser().parse_args(['run'])
        short = estimate_usage(args, 30)
        self.assertEqual(short['requests'], 90)
        self.assertGreater(short['cached_input_tokens'], 0)
        self.assertLess(short['cached_input_tokens'], short['input_tokens'])
        self.assertEqual(short['output_tokens'], 30 * 531)
        self.assertIsNone(short['reasoning_tokens'])
        self.assertEqual(short['total_tokens'], short['input_tokens'] + short['output_tokens'])
        args.input_tokens += 1000
        self.assertEqual(estimate_usage(args, 30)['input_tokens'] - short['input_tokens'], 60000)
        self.assertEqual(estimate_usage(args, 30)['cached_input_tokens'] - short['cached_input_tokens'], 30000)
        self.assertEqual(estimate_usage(args, 0)['total_tokens'], 0)
        self.assertEqual(estimate_usage(args, 0)['cached_input_tokens'], 0)
        args.output_tokens = 64
        self.assertLess(estimate_usage(args, 30)['output_tokens'], short['output_tokens'])

    def test_missing_total_is_not_reconstructed(self):
        sample_row = row('off', 0, 1, 100, valid=True, cached=100)
        del sample_row['turns'][0]['usage']['totalTokens']
        with self.assertRaises(KeyError):
            summarize([sample_row])

    def test_actual_usage_includes_invalid_samples_without_double_counting(self):
        a = row('off', 0, 1, 100, valid=False, cached=100)
        a['turns'] = [{'status': 'completed', 'stage': 'prefill', 'usage': {'inputTokens': 100, 'cachedInputTokens': 60,
                               'outputTokens': 10,
                               'reasoningOutputTokens': 3, 'totalTokens': 110}},
                      {'status': 'completed', 'stage': 'decode', 'usage': {'inputTokens': 200, 'cachedInputTokens': 100,
                                 'outputTokens': 20,
                                 'reasoningOutputTokens': 5, 'totalTokens': 220}}]
        totals = summarize([a])['reported_usage_totals']
        self.assertEqual(totals['inputTokens'], 300)
        self.assertEqual(totals['outputTokens'], 30)
        self.assertEqual(totals['totalTokens'], 330)
        output = render_console({'samples': [a], 'status': 'completed'})
        self.assertIn('total_tokens=330 complete=yes', output)
        a['turns'][0]['status'] = 'timeout'
        self.assertIn('complete=no', render_console({'samples': [a], 'status': 'completed'}))
        a['turns'][0]['status'] = 'completed'
        a['turns'].append({'usage': None, 'status': 'completed', 'stage': 'warmup'})
        summary = summarize([a])
        self.assertEqual(summary['requests_missing_usage'], 1)
        output = render_console({'samples': [a], 'status': 'completed'})
        self.assertIn('total_tokens=330 complete=no', output)
        a['turns'] = [{'usage': None, 'status': 'completed', 'stage': 'warmup'}]
        self.assertIn('total_tokens=N/A', render_console({'samples': [a], 'status': 'completed'}))
        self.assertIn('complete=no', render_console({'samples': [], 'status': 'interrupted'}))


class ReportTest(unittest.TestCase):
    def test_prefill_lengths_use_native_cached_and_input_tokens_of_valid_cases(self):
        a = row('off', 0, 1, 100, valid=True, cached=16000)
        b = row('off', 1, 2, 110, valid=True, cached=18000)
        invalid = row('off', 2, 3, 120, valid=False, cached=90000)
        for sample_row, count in ((a, 24000), (b, 26000), (invalid, 99000)):
            sample_row['turns'][0]['usage']['inputTokens'] = count
        summary = summarize([a, b, invalid])
        group = summary['groups'][0]
        self.assertEqual(group['prefill_cached_tokens'], distribution([16000, 18000]))
        self.assertEqual(group['prefill_input_tokens'], distribution([24000, 26000]))
        self.assertEqual(group['prefill_cached_tokens']['n'], group['ttft_s']['n'])
        output = render_console({'samples': [a, b, invalid], 'status': 'completed'})
        self.assertIn('Cached (tok)', output)
        self.assertIn('Input (tok)', output)
        self.assertIn('17000 ± 12706', output)
        self.assertIn('25000 ± 12706', output)
        self.assertEqual(group['output_tokens']['n'], 2)
        self.assertIn('533 ± 0', output)
        empty = summarize([invalid])['groups'][0]
        self.assertIsNone(empty['prefill_cached_tokens']['mean'])
        self.assertIsNone(empty['prefill_input_tokens']['mean'])

    def test_invalid_samples_contribute_neither_metric(self):
        a = row('off', 0, 4, 50, valid=False, cached=100)
        group = summarize([a])['groups'][0]
        self.assertEqual(group['ttft_s']['n'], 0)
        self.assertEqual(group['decode_tok_s']['n'], 0)
    def test_student_t_confidence_interval_and_small_samples(self):
        d = distribution([1, 2, 3, 4, 5])
        self.assertEqual(d['mean'], 3)
        self.assertAlmostEqual(d['ci95_half_width'], 1.963243, places=5)
        self.assertIsNone(distribution([2])['ci95'])
        self.assertIsNone(distribution([])['mean'])
        self.assertEqual(distribution([2, 2])['ci95_half_width'], 0)
        self.assertAlmostEqual(distribution([1, 3])['ci95_half_width'], 12.706205)
        data = {'samples': [row('on', 0, 1, 50, cached=100, valid=True)], 'status': 'completed', 'workload': {'input_tokens': 8192}}
        self.assertIn('± N/A', render_console(data))

    def test_models_never_share_statistics(self):
        a, b = row('off', 0, 4, 50, cached=100, valid=True), row('on', 0, 2, 100, cached=100, valid=True)
        b['model'] = b['effective']['model'] = 'other'
        summary = summarize([a, b])
        self.assertEqual(len(summary['groups']), 2)
        self.assertEqual([g['ttft_s']['mean'] for g in summary['groups']], [4, 2])


if __name__ == '__main__':
    unittest.main()
