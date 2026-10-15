#!/usr/bin/env python3
import json
import os
import re
import sys
import time
import tiktoken

if "--version" in sys.argv:
    print("fake-codex TEST DATA")
    raise SystemExit()

mode = os.environ.get("FAKE_CODEX_MODE", "ok")
turn_num = 0
previous_input = 0
previous_cached = 0
usage_total = {}
tier = None
model = "test-model"
effort = "medium"


def send(message):
    print(json.dumps(message), flush=True)


def event(method, params):
    send({"method": method, "params": {"threadId": "thread1", "turnId": f"turn{turn_num}", **params}})


for line in sys.stdin:
    request = json.loads(line)
    method, params = request.get("method"), request.get("params", {})
    result = {}
    if method == "initialized":
        continue
    if method == "initialize":
        result = {"userAgent": "fake-codex", "codexHome": "/fake/home"}
    elif method == "config/read":
        result = {"config": {"model": model, "model_reasoning_effort": effort}}
    elif method == "model/list":
        result = {"data": [{"model": model, "displayName": model, "defaultReasoningEffort": effort,
                            "isDefault": True, "supportedReasoningEfforts": [{"reasoningEffort": "medium"}]}], "nextCursor": None}
    elif method == "thread/start":
        tier = params.get("serviceTier")
        model = params.get("model", model)
        effort = params.get("config", {}).get("model_reasoning_effort", effort)
        result = {"thread": {"id": "thread1"}, "model": model, "modelProvider": "openai",
                  "reasoningEffort": effort, "serviceTier": tier, "approvalPolicy": "never",
                  "sandbox": {"type": "workspaceWrite"}}
        if mode == "ignore-tier":
            result.pop("serviceTier")
        if mode == "substitute-model":
            result["model"] = "different-model"
    elif method == "turn/start":
        turn_num += 1
        assert turn_num <= 3, "at most three turns per sample"
        prompt = params["input"][0]["text"]
        ack = "single letter K" in prompt
        turn = {"id": f"turn{turn_num}", "status": "inProgress"}
        event("turn/started", {"turn": turn})
        match = re.search(r'Copy these step IDs once, in this order: ([^\n]+)', prompt)
        ids = match.group(1).split() if match else [f'{n:02d}' for n in range(1, 1025)]
        body = ' '.join(['alpha beta gamma delta'] * 4)
        text = (("KK" if mode == "bad-ack" else "K") if ack else '\n'.join(
            f'{n}/{ids[-1]} {"DONE" if n == ids[-1] else "RUN"}: {body}' for n in ids) + '\nEND')
        first = text if mode == "single" else text[:5]
        event("item/agentMessage/delta", {"itemId": "a", "delta": first})
        send({"id": request["id"], "result": {"turn": turn}})
        if mode == "timeout":
            time.sleep(10)
        if mode == "disconnect":
            raise SystemExit()
        if mode == "interactive":
            send({"id": 900, "method": "item/tool/requestUserInput", "params": {}})
            continue
        if mode == "reroute":
            event("model/rerouted", {"fromModel": model, "toModel": "different-model"})
        if mode == "tool":
            event("item/started", {"item": {"id": "c", "type": "commandExecution"}})
        if mode not in ("short", "single"):
            rest = text[5:]
            for i in range(0, len(rest), 256):
                time.sleep(.0005)
                event("item/agentMessage/delta", {"itemId": "a", "delta": rest[i:i+256]})
        if mode == "hang-after-target" or (mode == "cache-miss-and-hang" and not ack):
            time.sleep(10)
        if mode != "no-usage":
            input_tokens = 16000 if turn_num == 1 else previous_input + len(tiktoken.get_encoding('o200k_base').encode(prompt)) + 64
            cached = previous_input if turn_num > 1 else 12000
            if mode in ('cache-miss', 'cache-miss-and-hang') and turn_num == 2:
                cached = 0
            if mode == 'decode-cache-miss' and turn_num == 3:
                cached = 0
            if mode == 'decode-cache-equal' and turn_num == 3:
                cached = previous_cached
            if mode == 'decode-cache-decreased' and turn_num == 3:
                cached = previous_cached - 128
            if mode == 'cache-boundary-gap':
                cached = previous_input - 352 if turn_num > 1 else 14080
            output_tokens = len(tiktoken.get_encoding('o200k_base').encode(text)) + 4
            last = {'inputTokens': input_tokens, 'cachedInputTokens': cached, 'cacheWriteInputTokens': 0,
                    'outputTokens': output_tokens, 'reasoningOutputTokens': 0, 'totalTokens': input_tokens + output_tokens}
            for key, value in last.items():
                usage_total[key] = usage_total.get(key, 0) + value
            event("thread/tokenUsage/updated", {"tokenUsage": {"total": usage_total, 'last': last}})
            previous_input = input_tokens
            previous_cached = cached
        turn["status"] = "failed" if mode == "failed" else "completed"
        event("turn/completed", {"turn": turn})
        continue
    send({"id": request["id"], "result": result})
