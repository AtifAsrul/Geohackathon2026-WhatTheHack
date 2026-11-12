"""The conversational assistant: Gemma (shared on the HPC) + evidence + tools.

The model cannot read a 3-D cube. So it is given:
  * a briefing of measured facts, each with an evidence id,
  * tools to look things up and to LOOK at pictures of sections and maps
    (Gemma can read images), and
  * strict rules on how a geoscientist argues: observation, interpretation,
    alternatives, confidence, recommendation - always citing evidence.
"""
import base64
import json
import re

import requests

from . import config, knowledge, tools

SYSTEM = """You are Exploration Copilot, an AI exploration geoscientist. You are working on ONE 3-D seismic survey \
with its wells. Your user is a geoscientist or a decision-maker who wants interpretations they can check.

WHAT YOU KNOW
You cannot read the raw seismic cube. You know (1) the BRIEFING below - facts measured from the data by an \
interpretation pipeline, each tagged with an evidence id such as [E12]; (2) the results of TOOLS you call; \
(3) pictures of the data when they are shown to you. The survey is anonymised: you do not know the basin, country, \
formation names or ages, so never invent them.

HOW TO REASON (think like a geoscientist)
For any interpretive question, structure the answer with these short bold headings:
**Observation** - what the data shows, with numbers. Cite the evidence id right after each fact, like [E12].
**Interpretation** - what it most likely means geologically, and why.
**Alternative** - at least one other explanation that fits the same observations, and what would tell them apart.
**Confidence** - low / medium / high, with the main reason, and what is missing.
**Recommendation** - the concrete next step (what to check, measure, or drill, and where).
For simple factual questions, just answer in one or two sentences - no headings.

RULES
- Every number or factual claim about this survey must come from the briefing, a tool result or a picture. If you do \
not have it, say so, or call a tool. Never make up well results, depths, or names.
- When you use general geological knowledge (not measured here), say "general principle".
- Scores in the briefing are simple transparent screening scores (0-1), not probabilities of success. Say so if asked.
- Be honest about limits listed under DATA QUALITY AND LIMITS.
- Plain language. Explain a technical word the first time in a few words. Be brief: usually under 250 words.
- To let the user jump to a place in the viewer, write a location tag exactly like \
[[IL 1234, XL 567, TWT 1500]] (optionally add ", vol=sand" using a volume role: full, near, mid, far, ufar, ai, si, \
vpvs, sand). Use one for each key place you mention.

TOOLS
If you need more detail than the briefing gives, reply with ONLY one line of JSON and nothing else, for example \
{"tool": "lead", "args": {"id": "L2"}}. You will get the result, then you may call another tool or answer. \
Use at most 4 tool calls per question. Never mention tools or JSON to the user.
- {"tool": "lead", "args": {"id": "L1"}}  -> full evidence for one lead
- {"tool": "well", "args": {"name": "p-1"}}  -> everything about one well
- {"tool": "horizon", "args": {"name": "H3"}}  -> structure, faults and wells on one horizon
- {"tool": "probe", "args": {"il": 1234, "xl": 567, "twt": 1500}}  -> all volume values, facies, horizons at a point
- {"tool": "look_section", "args": {"il": 1234, "twt": 1500, "vol": "full"}}  -> a PICTURE of an inline section \
(use "xl" instead of "il" for a crossline section); the viewer also jumps there
- {"tool": "look_map", "args": {"horizon": "H3", "attr": "time"}}  -> a PICTURE of a horizon map; attr is one of \
time, amp, rms_below, sand_below, sand_above, ai_below, vpvs_below, far_over_near, fault, facies, closure
- {"tool": "lookalike", "args": {"il": 1234, "xl": 567, "twt": 1500}}  -> other places whose seismic character \
resembles this spot (foundation-model features)
"""

NOT_READY = ("The interpretation has not been run yet, so I have no evidence to reason from. "
             "Click **Run interpretation** at the top, wait for it to finish, then ask me again.")


def _img(png):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


def _role_volume(role):
    if not role:
        return None
    from . import store
    for v in store.list_volumes():
        if v.meta["role"] == role or v.name == role:
            return v.name
    return None


def run_tool(name, args, view):
    """Returns (text result, optional PNG, optional viewer action)."""
    a = args or {}
    try:
        if name == "lead":
            return json.dumps(tools.lead(a.get("id")), default=str), None, None
        if name == "well":
            return json.dumps(tools.well(a.get("name")), default=str), None, None
        if name == "horizon":
            return json.dumps(tools.horizon(a.get("name")), default=str), None, None
        if name == "probe":
            return json.dumps(tools.probe(float(a["il"]), float(a["xl"]), float(a["twt"])), default=str), None, \
                {"il": a["il"], "xl": a["xl"], "twt": a["twt"]}
        if name == "lookalike":
            r = tools.lookalike(float(a["il"]), float(a["xl"]), float(a["twt"]))
            return json.dumps(r, default=str), None, {"il": a["il"], "xl": a["xl"], "twt": a["twt"], "layer": "similar"}
        if name == "look_section":
            twt = float(a["twt"]) if a.get("twt") is not None else None
            vol = _role_volume(a.get("vol")) or (view or {}).get("vol")
            if a.get("il") is not None:
                il, xl = float(a["il"]), float(a.get("xl") or (view or {}).get("xl") or 0)
                png = tools.render_section("i", il, vol, twt, (xl, twt) if twt is not None and a.get("xl") else None)
                act = {"il": il, "twt": twt, "axis": "i", "volrole": a.get("vol")}
                return f"Picture of inline {int(il)} attached. Horizons are the coloured lines with their names at the left.", png, act
            xl = float(a["xl"])
            png = tools.render_section("x", xl, vol, twt)
            return (f"Picture of crossline {int(xl)} attached. Horizons are the coloured lines with their names at the left.",
                    png, {"xl": xl, "twt": twt, "axis": "x", "volrole": a.get("vol")})
        if name == "look_map":
            png = tools.render_map(a.get("horizon"), a.get("attr", "time"))
            if png is None:
                return "No such map. Check the horizon name and attr.", None, None
            return (f"Picture of the {a.get('horizon')} map ({tools.MAP_LABEL.get(a.get('attr', 'time'))}) attached. "
                    "White outlines are leads (labelled), yellow dots are wells.", png,
                    {"map": {"horizon": a.get("horizon"), "attr": a.get("attr", "time")}})
        return f"Unknown tool '{name}'.", None, None
    except Exception as e:
        return f"The tool failed: {type(e).__name__}: {e}. Check the arguments.", None, None


def _step_text(name, a):
    a = a or {}
    return {
        "lead": f"Reading the evidence for {a.get('id', 'a lead')}",
        "well": f"Checking well {a.get('name', '')}",
        "horizon": f"Checking horizon {a.get('name', '')}",
        "probe": f"Reading the data at IL {a.get('il')} / XL {a.get('xl')} / {a.get('twt')} ms",
        "lookalike": f"Searching for places that look like IL {a.get('il')} / XL {a.get('xl')} / {a.get('twt')} ms",
        "look_section": f"Looking at {'inline ' + str(a.get('il')) if a.get('il') is not None else 'crossline ' + str(a.get('xl'))}",
        "look_map": f"Looking at the {a.get('horizon', '')} {a.get('attr', 'time')} map",
    }.get(name, f"Using {name}")


def _parse_tool(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    if not t.startswith("{"):
        return None
    try:
        obj = json.loads(t[:t.rindex("}") + 1])
    except Exception:
        return None
    if isinstance(obj, dict) and "tool" in obj:
        return obj
    return None


def _visible(raw):
    """Drop any private 'thinking' block a reasoning model may put before its answer."""
    out = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    cut = out.find("<think>")
    return out if cut < 0 else out[:cut]


def _stream(messages, max_tokens=2000, temperature=0.3):
    """Yield text pieces from the OpenAI-compatible endpoint."""
    if config.LLM_URL == "mock":
        yield from _mock(messages)
        return
    r = requests.post(config.LLM_URL.rstrip("/") + "/chat/completions", timeout=(10, 300), stream=True,
                      json={"model": config.LLM_MODEL, "messages": messages, "temperature": temperature,
                            "max_tokens": max_tokens, "stream": True})
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            delta = json.loads(data)["choices"][0].get("delta", {})
        except Exception:
            continue
        piece = delta.get("content")
        if piece:
            yield piece


def _mock(messages):
    """Stand-in model for testing the app without the HPC."""
    last = messages[-1]["content"]
    last = last if isinstance(last, str) else " ".join(p.get("text", "") for p in last)
    if "TOOL RESULT" not in last and "lead" in last.lower():
        yield '{"tool": "lead", "args": {"id": "L1"}}'
        return
    k = knowledge.load() or {}
    l = (k.get("leads") or [{}])[0]
    w = l.get("where", {"il": 0, "xl": 0})
    for piece in re.split(r"(\s+)", f"**Observation** (test mode, no real model) The top lead is {l.get('id')} at {l.get('horizon')} "
                                    f"[{(l.get('for') or ['E1'])[0]}]. Jump there: [[IL {w.get('il')}, XL {w.get('xl')}, TWT {l.get('crest_twt_ms', 0)}]].\n\n"
                                    "**Interpretation** This text comes from the built-in stand-in.\n\n**Alternative** n/a\n\n"
                                    "**Confidence** n/a\n\n**Recommendation** Run on the HPC to talk to Gemma."):
        yield piece


def _view_note(view):
    if not view:
        return ""
    try:
        bits = [f"inline {view['il']}", f"crossline {view['xl']}", f"TWT {view['twt']} ms", f"volume {view.get('vol_label', view.get('vol'))}"]
        p = tools.probe(float(view["il"]), float(view["xl"]), float(view["twt"])) if tools.grid() is not None else {}
        if p.get("between_horizons"):
            bits.append(f"between {p['between_horizons'][0]} and {p['between_horizons'][1]}")
        if p.get("inside_leads"):
            bits.append("inside " + ", ".join(p["inside_leads"]))
        return "\n\n(The user's crosshair is at: " + ", ".join(bits) + ".)"
    except Exception:
        return ""


def chat(history, view=None, attach_view=False):
    """Generator of events: {'type': 'step'|'view'|'token'|'error'|'done', ...}."""
    k = knowledge.load()
    if not k or not tools.ready():
        yield {"type": "token", "text": NOT_READY}
        yield {"type": "done"}
        return
    hist = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")][-9:]
    while hist and hist[0]["role"] != "user":
        hist = hist[1:]
    if not hist:
        yield {"type": "done"}
        return
    last = dict(hist[-1])
    content = [{"type": "text", "text": last["content"] + _view_note(view)}]
    if attach_view and view:
        try:
            ax = view.get("axis", "i")
            num = float(view["il"] if ax == "i" else view["xl"])
            mark = (float(view["xl"] if ax == "i" else view["il"]), float(view["twt"]))
            content.append(_img(tools.render_section(ax, num, view.get("vol"), float(view["twt"]), mark)))
            content[0]["text"] += ("\n(A picture of the section the user is looking at is attached; the cyan circle "
                                   "marks the crosshair.)")
        except Exception as e:
            yield {"type": "step", "text": f"Could not attach the view ({e})"}
    hist[-1] = {"role": "user", "content": content if len(content) > 1 else content[0]["text"]}

    def build(compact):
        brief = knowledge.briefing(k, max_leads=6, compact=True) if compact else k["briefing"]
        h = hist[-3:] if compact else hist
        while h and h[0]["role"] != "user":
            h = h[1:]
        return [{"role": "system", "content": SYSTEM + "\nBRIEFING\n" + brief}] + h

    messages = build(False)
    compacted = False
    for _round in range(6):
        buf, mode, raw, sent = "", None, "", 0
        try:
            for piece in _stream(messages):
                raw += piece
                buf = _visible(raw)
                if mode is None:
                    head = buf.lstrip()
                    if len(head) >= 8 or "\n" in head:
                        mode = "tool" if head.startswith(("{", "```")) else "answer"
                if mode == "answer" and len(buf) > sent:
                    yield {"type": "token", "text": buf[sent:]}
                    sent = len(buf)
        except Exception as e:
            msg = str(e)
            if not compacted and ("HTTP 400" in msg or "context" in msg.lower()):
                compacted = True          # too long for the model, or it dislikes a system message: shrink and fold
                messages = build(True)
                sys_text = messages.pop(0)["content"]
                first = messages[0]
                if isinstance(first["content"], str):
                    first["content"] = sys_text + "\n\nUSER QUESTION\n" + first["content"]
                else:
                    first["content"][0]["text"] = sys_text + "\n\nUSER QUESTION\n" + first["content"][0]["text"]
                continue
            yield {"type": "error", "text": f"The language model could not be reached ({msg[:300]}). "
                                            f"Check that the app is running on the HPC and that {config.LLM_URL} is up."}
            yield {"type": "done"}
            return
        call = _parse_tool(buf) if mode != "answer" else None
        if call is None and mode == "answer":
            # the model wrote a few words and THEN a tool request: honour it, and take the text back
            tail = next((ln for ln in reversed(buf.strip().splitlines()) if ln.strip()), "")
            call = _parse_tool(tail) if '"tool"' in tail else None
            if call is not None:
                yield {"type": "reset"}
                buf = tail
        if call is None:
            if mode != "answer" and buf.strip():
                yield {"type": "token", "text": buf}
            yield {"type": "done"}
            return
        if _round >= 4:
            messages += [{"role": "assistant", "content": buf},
                         {"role": "user", "content": "No more tool calls are allowed. Answer the user now with what you have."}]
            continue
        name, args = call.get("tool"), call.get("args") or {}
        yield {"type": "step", "text": _step_text(name, args)}
        result, png, action = run_tool(name, args, view)
        if action:
            yield {"type": "view", "action": action}
        body = [{"type": "text", "text": f"TOOL RESULT ({name}):\n{result[:6000]}\n\nNow either call another tool "
                                         "(one line of JSON only) or answer the user's question."}]
        if png:
            body.append(_img(png))
        messages += [{"role": "assistant", "content": buf.strip()},
                     {"role": "user", "content": body if png else body[0]["text"]}]
    yield {"type": "done"}


def check():
    """Is the language model reachable? Returns (ok, message)."""
    if config.LLM_URL == "mock":
        return True, "test stand-in (no real model)"
    try:
        r = requests.get(config.LLM_URL.rstrip("/") + "/models", timeout=5)
        if r.status_code == 200:
            ids = [m.get("id") for m in r.json().get("data", [])]
            return True, ", ".join(ids) or "connected"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, type(e).__name__
