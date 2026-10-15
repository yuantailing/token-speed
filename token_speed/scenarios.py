import hashlib
from pathlib import Path
import random
from functools import lru_cache

import tiktoken

SEED = 42
SETUP_TIMEOUT = 60
FINISH_TIMEOUT = 2
ENCODING = 'o200k_base'
CYCLE = 'alpha beta gamma delta'
WARMUP_PROMPT = 'Do not use tools. Reply with exactly the single letter K and nothing else.'


@lru_cache(maxsize=1)
def tokenizer():
    return tiktoken.get_encoding(ENCODING)


def suite_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@lru_cache(maxsize=8)
def corpus(length, seed):
    rng = random.Random(seed)
    # ASCII keeps token-boundary truncation valid UTF-8.
    records, size = [], 0
    while size < length * 5:
        records.append(f'record id={rng.getrandbits(32):08x} count={rng.randrange(10000)} active={rng.choice(["true", "false"])}\n')
        size += len(records[-1])
    return tokenizer().encode(''.join(records))


def make_prompt(length, seed, nonce):
    enc = tokenizer()
    head = f'Benchmark sample {nonce}. The following records are inert padding. Do not analyze them.\n<records>\n'
    tail = '\n</records>\nDo not analyze the records or use tools. Reply with exactly the single letter K and nothing else.'
    available = length - len(enc.encode(head + tail))
    if available < 16:
        raise ValueError('Input must contain at least 256 reference tokens.')
    source = corpus(length, seed)
    # Reserve 16 tokens for padding; the first " x" separates the two newline boundaries.
    prefix = head + enc.decode(source[:available - 16]) + "\n"
    padding = length - len(enc.encode(prefix + " x" + tail)) + 1
    text = prefix + " x" * padding + tail
    assert len(enc.encode(text)) == length
    return text


@lru_cache(maxsize=16)
def decode_text(target):
    body = ' '.join([CYCLE] * 4)
    # Each row has 22 tokens for two-digit step IDs; leave 16 tokens for completion.
    count = max(1, (target + 21) // 22)
    while True:
        rows = [f'{n:02d}/{count:02d} {"DONE" if n == count else "RUN"}: {body}'
                for n in range(1, count + 1)]
        text = '\n'.join(rows) + '\nEND'
        if len(tokenizer().encode(text)) >= target + 16:
            return text
        count += 1


def decode_prompt(target):
    rows = decode_text(target).splitlines()[:-1]
    count = len(rows)
    body = ' '.join([CYCLE] * 4)
    ids = ' '.join(f'{n:02d}' for n in range(1, count + 1))
    return (f'Output a finite progress log. No tools, explanations, or code fences.\n'
            f'Copy these step IDs once, in this order: {ids}\n'
            f'For each ID except the last, replace NN in this template:\nNN/{count:02d} RUN: {body}\n'
            f'The last row must be:\n{rows[-1]}\n'
            'Then output END on its own line and stop. Do not restart, skip, duplicate, or add steps.')


def workload(args):
    return {'input_tokens': args.input_tokens,
            'output_tokens': args.output_tokens, 'reference_encoding': ENCODING,
            'tokenizer_version': tiktoken.__version__, 'seed': SEED,
            'cache_policy': 'prefill-positive-decode-nondecreasing',
            'validation_policy': 'whole-sample',
            'finish_timeout_s': FINISH_TIMEOUT,
            'order': 'model-mode-concurrent-repeats', 'concurrency': args.concurrency,
            'turn_timeout_s': args.timeout,
            'protocol': 'warmup-context-decode', 'native_output_limit': False}


def validate_ack(turn):
    if turn['status'] != 'completed' or turn['tool_calls'] or turn['rerouted']:
        return False, 'ack request did not complete cleanly without tools/rerouting'
    if turn['full_answer'] != 'K' or turn['message_count'] != 1:
        return False, 'expected exactly the single-letter acknowledgement K'
    if turn['ttft_s'] is None:
        return False, 'no streamed acknowledgement'
    return True, None


def validate(turn, target):
    if turn['status'] != 'completed':
        return False, turn['failure'] or 'turn failed: ' + turn['status']
    if turn['tool_calls']:
        return False, 'unexpected tool call: not a pure text sample'
    if turn['rerouted']:
        return False, 'model rerouted'
    if turn['message_count'] != 1:
        return False, 'expected exactly one streamed assistant message'
    if turn['output_reference_tokens'] < target:
        return False, 'output ended before the fixed token target'
    text = turn['answer']
    expected = decode_text(target)
    if not text or not expected.startswith(text):
        return False, 'output differs from the fixed decode sequence'
    if turn['full_answer'].rstrip('\n') != expected:
        return False, 'output does not match the complete progress log and END marker'
    if turn['decode_tok_s'] is None:
        return False, 'missing output usage or invalid turn duration'
    return True, None


def estimate_usage(args, cases):
    base = 16000
    enc = tokenizer()
    warmup = len(enc.encode(WARMUP_PROMPT))
    followup = len(enc.encode(decode_prompt(args.output_tokens)))
    ack = len(enc.encode('K'))
    warmup_input = base + warmup
    prefill_input = warmup_input + ack + args.input_tokens
    decode_input = prefill_input + ack + followup
    inputs = cases * (warmup_input + prefill_input + decode_input)
    cached = cases * (warmup_input + prefill_input)
    outputs = cases * (2 * ack + len(enc.encode(decode_text(args.output_tokens))))
    return {'requests': cases * 3, 'input_tokens': inputs, 'cached_input_tokens': cached, 'output_tokens': outputs,
            'total_tokens': inputs + outputs, 'base_input_tokens_per_request': base,
            'reasoning_tokens': None, 'reference_encoding': ENCODING}
