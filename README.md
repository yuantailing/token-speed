# Token Speed

Measure prefill latency and decode throughput through the local `codex app-server`, preserving the default system prompt and local Codex configuration.

## Usage

Requires Python 3.11+ and Codex app-server.

```bash
python3 -m pip install -e .
token-speed run
```

The module entry point is `python3 -m token_speed`. Use `token-speed doctor` to inspect configuration and the model catalog without inference.

| Option | Default / description |
| --- | --- |
| `--input-tokens` | 32768; one input length per run, minimum 256 |
| `--output-tokens` | 512; decode measurement target, minimum 64 |
| `--timeout` | 30; inference timeout per request, in seconds |
| `--model` | All available models among gpt-6.1-sol, gpt-6-astra, gpt-6-luna; repeatable |
| `--effort` | low |
| `--fast` | both; accepts on, off, inherit |
| `--repeats` | 5 per model/mode; 2 for input lengths ≥ 65536 |
| `--concurrency` | 5 within each model/mode group; 1 for sequential execution |
| `--yes` | Skip token usage confirmation |
| `--json` | Write results to a new JSON file |
| `--profile` | Codex configuration profile |
| `--config` | Codex `key=value` override; repeatable |
| `--codex` | Executable path; defaults to CODEX_BIN or codex on PATH |

Model/mode groups execute sequentially. Each completed case produces one table row; the final summary reports means ± 95% confidence intervals.

Estimated usage is displayed before confirmation; actual usage includes invalid cases. Cache estimates assume an uncached warmup and reuse of the preceding request's input in each subsequent request, with a 16k base input allowance per request and no reasoning allowance. Cached tokens are included in input tokens, and reasoning tokens in output tokens; neither is added again to total tokens. Incomplete usage is marked `complete=no`.

## Measurement

Each case uses a separate session for three requests: warmup, the specified context with a single-letter acknowledgement, then a short continuation requesting a finite progress log. A random prefix prevents reuse of the long input across cases.

- Input and output targets use `o200k_base`; the system prompt and conversation history are additional input.
- Summary `Input (tok)` and `Cached (tok)` report the prefill request's full input and cached prefix in native tokens, using the same valid samples as the speed metrics.
- Prefill latency measures submission to the first text fragment, including network, queueing, and reasoning time.
- `Output (tok)` reports the decode request's native `outputTokens`. Decode throughput divides this count by the elapsed time from the first text fragment to the final `turn/completed` event.
- Prefill cache usage must be positive; decode cache usage must be at least as large. Any validation failure excludes the entire case from summary statistics.
- Output length is controlled by the prompt and client measurement window, not a server token limit. Completion has a two-second grace period after the target is reached, within the request timeout. Failed cases are not retried; Ctrl-C cancels execution.

## Performance reference

Approximate measurements from 2026-10-15 with 32768 input tokens, 512 output tokens, low effort, Fast ON, and concurrency 5. These indicate scale; measure the target configuration for precise results.

| Model | Prefill | Decode |
| --- | --- | --- |
| gpt-6.1-sol | ~2 s | ~80 tokens/s |
| gpt-6-astra | ~3 s | ~55 tokens/s |
| gpt-6-luna | ~2 s | ~150 tokens/s |

## Tests

`python3 -m unittest discover -s tests -v`
