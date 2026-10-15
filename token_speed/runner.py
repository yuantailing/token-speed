from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby
import threading
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import uuid

from .metrics import run_turn, TOKEN_KEYS
from .rpc import AppServer, ProtocolError
from .scenarios import (SEED, SETUP_TIMEOUT,
                        FINISH_TIMEOUT, WARMUP_PROMPT, make_prompt, tokenizer, validate,
                        validate_ack, decode_prompt)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def command(args, fast):
    cmd = [args.codex, "app-server"]
    for override in args.config:
        cmd += ["-c", override]
    if args.profile:
        cmd += ["-c", "profile=" + json.dumps(args.profile)]
    if fast == "on":
        cmd += ["-c", "features.fast_mode=true"]
    return cmd


def read_environment(server, cwd):
    config = server.call("config/read", {"includeLayers": False, "cwd": str(cwd)})["config"]
    provider = config.get("model_provider") or "openai"
    provider_config = (config.get("model_providers") or {}).get(provider, {})
    routing = {k: provider_config.get(k) for k in (
        "base_url", "wire_api", "env_key", "requires_openai_auth", "supports_websockets")}
    routing["network_environment"] = {k: os.environ.get(k) for k in (
        "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY", "https_proxy", "http_proxy",
        "all_proxy", "no_proxy", "OPENAI_BASE_URL", "CODEX_CA_CERTIFICATE", "SSL_CERT_FILE")}
    result = {
        "provider": provider,
        "route_fingerprint": digest(routing),
        "context_fingerprint": digest({
            **{k: config.get(k) for k in ("mcp_servers", "model_instructions_file", "developer_instructions", "tools")},
            # Feature overrides materialize unspecified values as null.
            "features": {k: v for k, v in (config.get("features") or {}).items()
                         if k != "fast_mode" and v is not None},
        }),
        "configured_model": config.get("model"),
        "configured_effort": config.get("model_reasoning_effort"),
        "configured_service_tier": config.get("service_tier"),
        "platform": platform.system() + "/" + platform.machine(),
        "python": platform.python_version(),
    }
    return result


def inspect(args):
    with AppServer(command(args, "inherit"), str(Path.cwd()), SETUP_TIMEOUT, threading.Event()) as server:
        info = read_environment(server, Path.cwd())
        info["server"] = server.info
        models, cursor = [], None
        while True:
            params = {"includeHidden": True, "limit": 100}
            if cursor:
                params["cursor"] = cursor
            page = server.call("model/list", params)
            models += [{k: model[k] for k in ("model", "displayName", "defaultReasoningEffort",
                       "supportedReasoningEfforts", "isDefault")} for model in page["data"]]
            cursor = page.get("nextCursor")
            if not cursor:
                break
        info["measurement_capabilities"] = {"native_max_output_tokens": False, "native_session_output_limit": False, "explicit_cache_control": False, "per_segment_cache_attribution": False}
        info["models"] = models
        return info


def verify_settings(response, model, effort, fast):
    if response["model"] != model:
        raise ProtocolError("Codex resolved a different model than requested")
    if response["reasoningEffort"] != effort:
        raise ProtocolError("Codex did not confirm the requested effort")
    if fast != "inherit":
        tier = response["serviceTier"]
        if fast == "on" and tier not in ("fast", "priority"):
            raise ProtocolError("Codex did not confirm Fast mode")
        if fast == "off" and tier not in (None, "default"):
            raise ProtocolError("Codex did not confirm Standard mode")


def same_environment(a, b, *, check_tier):
    ea, eb = a["effective"], b["effective"]
    keys = ["model", "modelProvider", "reasoningEffort", "approvalPolicy", "sandbox"]
    if check_tier:
        keys.append("serviceTier")
    if any(ea[k] != eb[k] for k in keys):
        return False
    for k in ("route_fingerprint", "context_fingerprint", "platform", "python"):
        if a["environment"][k] != b["environment"][k]:
            return False
    return a["instruction_fingerprint"] == b["instruction_fingerprint"]


def sample(args, model, fast, repeat, cancel):
    length = args.input_tokens
    nonce = uuid.uuid4().hex
    prompt = make_prompt(length, SEED, nonce)
    row = {"model": model, "fast": fast, "repeat": repeat,
           "valid": False, "turns": [], "failure": None, "effort": args.effort,
           "effective": None, "environment": None, "instruction_fingerprint": None,
           "input_reference_tokens": len(tokenizer().encode(prompt)),
           "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
           "nonce": nonce, "cache_checks": {}, "native_output_limit": False}
    try:
        with tempfile.TemporaryDirectory(prefix="token-speed-") as temp:
            root = Path(temp)
            (root / ".git").mkdir()  # Stop parent project discovery.
            setup_start = time.perf_counter()
            with AppServer(command(args, fast), str(root), SETUP_TIMEOUT,
                           cancel_event=cancel) as server:
                row["environment"] = read_environment(server, root)
                params = {"cwd": str(root), "ephemeral": True, "approvalPolicy": "never",
                          "sandbox": "read-only", "model": model,
                          "config": {"model_reasoning_effort": args.effort}}
                if fast != "inherit":
                    params["serviceTier"] = "fast" if fast == "on" else "default"
                response = server.call("thread/start", params)
                row["setup_s"] = time.perf_counter() - setup_start
                row["effective"] = {k: response[k] for k in (
                    "model", "modelProvider", "reasoningEffort", "serviceTier", "approvalPolicy")}
                row["effective"]["sandbox"] = response["sandbox"]["type"]
                row["instruction_fingerprint"] = digest([
                    hashlib.sha256(Path(p).read_bytes()).hexdigest()
                    for p in response.get("instructionSources", [])])
                verify_settings(response, model, args.effort, fast)
                row["setup_warnings"] = [m["method"] for _, m in server.pending
                                         if m["method"] in ("warning", "configWarning")]
                baseline = dict.fromkeys(TOKEN_KEYS, 0)
                failures = []
                stages = [('warmup', WARMUP_PROMPT, 1), ('prefill', prompt, 1),
                          ('decode', decode_prompt(args.output_tokens), args.output_tokens)]
                for stage, text, target in stages:
                    turn = run_turn(server, response['thread']['id'], text, args.timeout, target, baseline,
                                    finish_timeout=FINISH_TIMEOUT)
                    turn['stage'] = stage
                    turn['submitted_reference_tokens'] = len(tokenizer().encode(text))
                    row['turns'].append(turn)
                    baseline = turn['cumulative_usage']
                    ok, why = validate(turn, target) if stage == 'decode' else validate_ack(turn)
                    if not ok:
                        failures.append(f'{stage}: {why}')
                        break
                    if stage != 'warmup':
                        previous = row['turns'][-2]
                        evidence = cache_evidence(previous, turn, require_nondecreasing=(stage == 'decode'))
                        row['cache_checks'][stage] = evidence
                        if not evidence['passed']:
                            cached, prior = evidence['cached_input_tokens'], evidence['previous_cached_input_tokens']
                            reason = f'{stage}: cache check failed (cached={cached}, previous_cached={prior})'
                            failures.append(reason)
                    if baseline is None:
                        failures.append(f'{stage}: missing token usage')
                        break
                row['failure'] = '; '.join(failures) or None
                row['valid'] = not failures
    except (ProtocolError, TimeoutError, EOFError, InterruptedError) as exc:
        row["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        for turn in row["turns"]:
            del turn["answer"], turn["full_answer"]
    return row


def cache_evidence(previous, current, require_nondecreasing):
    prior = previous['usage']['cachedInputTokens'] if previous['usage'] is not None else None
    cached = current['usage']['cachedInputTokens'] if current['usage'] is not None else None
    positive = type(cached) is int and cached > 0
    nondecreasing = type(prior) is int and prior >= 0 and positive and cached >= prior
    return {'passed': positive and (not require_nondecreasing or nondecreasing),
            'previous_cached_input_tokens': prior, 'cached_input_tokens': cached,
            'positive_cache': positive, 'cache_nondecreasing': nondecreasing,
            'nondecreasing_required': require_nondecreasing,
            'segment_attribution_available': False}


def schedule(models, fast, repeats):
    modes = ['off', 'on'] if fast == 'both' else [fast]
    for model in models:
        for mode in modes:
            for repeat in range(repeats):
                yield model, mode, repeat


def run_samples(args, plan, run_case):
    cancel = threading.Event()
    pool = ThreadPoolExecutor(max_workers=args.concurrency)
    pending = {}
    try:
        for _, group in groupby(enumerate(plan, 1), key=lambda entry: entry[1][:2]):
            pending = {}
            for index, (model, fast, repeat) in group:
                future = pool.submit(run_case, args, model, fast, repeat, cancel)
                pending[future] = index
            for future in as_completed(pending):
                yield pending[future], future.result()
    finally:
        cancel.set()
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
