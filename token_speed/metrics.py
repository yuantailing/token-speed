import time

from .rpc import ProtocolError
from .scenarios import tokenizer

TOKEN_KEYS = ('inputTokens', 'cachedInputTokens', 'outputTokens', 'reasoningOutputTokens', 'totalTokens')
NON_TOOL_TYPES = {'userMessage', 'agentMessage', 'reasoning', 'plan'}


class TurnMetrics:
    def __init__(self, thread_id, turn_id, start, target, baseline):
        self.thread_id, self.turn_id, self.start = thread_id, turn_id, start
        self.target = target
        self.baseline = baseline
        self.cumulative_usage = None
        self.full_text = ""
        self.first = self.last = self.end = None
        self.text = ''
        self.message_ids = set()
        self.delta_count = self.tool_calls = 0
        self.usage = None
        self.status = 'inProgress'
        self.failure = None
        self.rerouted = self.target_reached = False
        self.warnings, self.trace = [], []

    def event(self, stamp, message):
        p, method = message['params'], message['method']
        if p.get('threadId') != self.thread_id:
            return
        event_turn = p['turn']['id'] if method in ('turn/started', 'turn/completed') else p.get('turnId')
        if event_turn and event_turn != self.turn_id:
            return
        t = max(0., stamp - self.start)
        item = p.get('item', {})
        if method == 'item/agentMessage/delta' and p['delta']:
            self.full_text += p['delta']
            self.message_ids.add(p['itemId'])
            if not self.target_reached:
                self.text += p['delta']
                self.delta_count += 1
                if self.first is None:
                    self.first = t
                self.last = t
                self.target_reached = len(tokenizer().encode(self.text)) >= self.target
        elif method == 'item/started' and p['item']['type'] not in NON_TOOL_TYPES:
            self.tool_calls += 1
        elif method == 'thread/tokenUsage/updated':
            total = p['tokenUsage']['total']
            self.cumulative_usage = {key: total[key] for key in TOKEN_KEYS}
            self.usage = {key: total[key] - self.baseline[key] for key in TOKEN_KEYS}
            if any(value < 0 for value in self.usage.values()):
                raise ValueError('Cumulative token usage decreased')
        elif method == 'model/rerouted':
            self.rerouted = True
        elif method in ('error', 'warning', 'model/verification'):
            self.warnings.append(method)
        elif method == 'turn/completed':
            self.end, self.status = stamp, p['turn']['status']
        if method in ('item/started', 'item/completed', 'turn/started', 'turn/completed',
                      'model/rerouted', 'error', 'warning', 'thread/tokenUsage/updated'):
            self.trace.append({'s': round(t, 6), 'event': method, 'item_type': item.get('type')})

    def result(self):
        count = len(tokenizer().encode(self.text))
        elapsed = (self.end or time.perf_counter()) - self.start
        output = self.usage['outputTokens'] if self.usage is not None else None
        seconds = elapsed - self.first if self.end is not None and self.first is not None else None
        rate = output / seconds if self.status == 'completed' and output is not None and seconds is not None and seconds > 0 else None
        return {'status': self.status, 'failure': self.failure, 'elapsed_s': elapsed,
                'ttft_s': self.first, 'prefill_server_s': None,
                'decode_s': seconds, 'decode_tok_s': rate, 'output_tokens': output,
                'output_reference_tokens': count, 'stream_deltas': self.delta_count,
                'target_reached': self.target_reached, 'overshoot_reference_tokens': max(0, count-self.target),
                'message_count': len(self.message_ids), 'tool_calls': self.tool_calls,
                'usage': self.usage, 'cumulative_usage': self.cumulative_usage, 'rerouted': self.rerouted, 'warnings': self.warnings,
                'timeline': self.trace, 'answer': self.text, 'full_answer': self.full_text}


def run_turn(server, thread_id, prompt, timeout, target, baseline, *, finish_timeout):
    tokenizer()
    buffered = []
    start = time.perf_counter()
    metrics = TurnMetrics(thread_id, None, start, target, baseline)
    draining = False
    timeout_phase = None
    deadline = start + timeout
    try:
        request = server.request('turn/start', {
            'threadId': thread_id, 'input': [{'type': 'text', 'text': prompt}],
        })
        response = server.response(request, deadline, buffered)
        metrics.turn_id = response['turn']['id']
        for stamp, message in buffered:
            metrics.event(stamp, message)
        while metrics.end is None:
            if metrics.target_reached and not draining:
                draining = True
                deadline = min(deadline, start + metrics.last + finish_timeout)
            stamp, message = server.receive(deadline)
            metrics.event(stamp, message)
    except TimeoutError:
        metrics.status = 'timeout'
        timeout_phase = 'completion' if draining else 'request'
        if metrics.turn_id:
            server.request('turn/interrupt', {'threadId': thread_id, 'turnId': metrics.turn_id})
    except (EOFError, ProtocolError, BrokenPipeError) as exc:
        metrics.status = 'transport_error'
        metrics.failure = str(exc)
    result = metrics.result()
    if timeout_phase:
        result['timeout_phase'] = timeout_phase
        result['failure'] = (f'completion timeout after {len(tokenizer().encode(metrics.text))} measured tokens'
                             if timeout_phase == 'completion' else 'request timeout before token target')
    return result
