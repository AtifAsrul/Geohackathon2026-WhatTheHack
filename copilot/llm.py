"""The conversational assistant: Gemma (shared on the HPC) + evidence + tools.

The model cannot read a 3-D cube. So it is given:
  * a briefing of measured facts, each with an evidence id,
  * tools to look things up and to LOOK at pictures of sections and maps
    (Gemma can read images), and
  * strict rules on how a geoscientist argues: observation, interpretation,
    alternatives, confidence, recommendation - always citing evidence.
"""
import base64
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from . import config, knowledge, tools

SYSTEM = """You are Exploration Copilot, an AI exploration geoscientist. You are working on ONE 3-D seismic survey \
with its wells. Your user is a geoscientist or a decision-maker who wants interpretations they can check.

WHAT YOU KNOW
You cannot read the raw seismic cube. You know (1) the BRIEFING below - facts from an interpretation pipeline, each \
tagged with an evidence id such as [E12]; (2) the results of TOOLS you call; (3) pictures of the data when they are \
shown to you. The survey is anonymised: you do not know the basin, country, formation names or ages, so never invent them.

KINDS OF EVIDENCE - keep them apart when you argue
- MEASURED: numbers computed from the seismic volumes (horizon times, closures, amplitudes, sand probability, coherence).
- WELLS: what the logs show, and checks of the seismic against the logs (well ties, rock certainty). Strongest evidence.
- ASSISTANT'S OWN READINGS: what a vision model (you, in an earlier pass) saw in pictures: time-slice features and \
the second-opinion horizons G1, G2... Fault sticks are algorithmic measured results; Gemma never selects or draws them. \
These visual readings are interpretations, weaker than measurements. Say "the visual \
reading suggests" and quote the measured check that comes with each.
- Two horizon sets exist: H# from RGT-Est (used for the traps) and G# picked on a net of lines. Where they agree the \
structure is robust; where they differ, say so.
- "Rock certainty" (e.g. sand 0.78) comes from testing the sand prediction on the wells, each well predicted from the \
others. Quote the basis that comes with it; never state a certainty the tools did not give you.

HOW YOU WORK ON A QUESTION
1. Locate: where is the user (inline, crossline, time), between which horizons, in which lead, near which well and fault. \
The app tells you this in brackets after the question.
2. Gather: use the briefing first; call a tool when you need detail or need to look.
3. Reason: apply geology and geophysics to what you gathered.
4. State uncertainty: an alternative reading, your confidence, and what is missing.
5. Recommend: the concrete next step.

HOW TO ANSWER (think like a geoscientist)
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
vpvs, sand, coh). Use one for each key place you mention.

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
- {"tool": "look_slice", "args": {"twt": 1500, "vol": "full"}}  -> a PICTURE of a time slice (map view): the volume on \
the left, coherence on the right, wells and named faults marked. Use it for channels, fault patterns, fold shapes.
- {"tool": "look_well", "args": {"name": "p-1"}}  -> a PICTURE of the well's logs plus its sand / pay zones
- {"tool": "fault", "args": {"id": "F1"}}  -> one named fault: strike, length, throw, and what was seen on each stick
- {"tool": "compare_horizons", "args": {"name": "G2"}}  -> how the second-opinion horizon agrees with RGT (omit name for all)
- {"tool": "dossier", "args": {"lead": "L1"}}  -> everything tied together for one lead; or use \
{"il": 1234, "xl": 567, "twt": 1500} for any place: rock certainty, nearest well logs, nearest fault, both horizon sets
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
        if name == "look_slice":
            twt = float(a["twt"])
            role = a.get("vol") if a.get("vol") in ("full", "sand", "ai", "vpvs", "coh", "near", "far") else "full"
            if tools.gcube(role) is None:
                role = "full"
            mark = (float(view["il"]), float(view["xl"])) if view and view.get("il") is not None else None
            png = tools.render_slice(twt, role, mark=mark)
            return (f"Picture of the time slice at {twt:.0f} ms attached (map view: crossline left to right, inline top to bottom). "
                    "Left: the volume. Right: coherence (dark = faults or channel edges). Yellow dots are wells, red lines are named "
                    "faults, the cyan mark is the user's crosshair.", png, {"twt": twt, "mapmode": "slice", "volrole": a.get("vol")})
        if name == "look_well":
            png = tools.render_well(a.get("name"))
            d = tools.well(a.get("name"))
            if png is None:
                return json.dumps(d, default=str), None, None
            keep = {q: d.get(q) for q in ("name", "il", "xl", "td_source", "zones", "assistant_reading", "horizons", "notes")}
            return ("Picture of the well's logs attached (yellow = sand, red = sand with a hydrocarbon flag; coloured lines are "
                    "horizons). Measured zones: " + json.dumps(keep, default=str)), png, \
                ({"il": d["il"], "xl": d["xl"], "well": d["name"]} if d.get("il") is not None else None)
        if name == "fault":
            d = tools.fault(a.get("id"))
            return json.dumps(d, default=str), None, d.get("where")
        if name == "compare_horizons":
            d = tools.compare_horizons(a.get("name"))
            act = d.get("largest_difference") if isinstance(d, dict) else None
            png = None
            if act:
                png = tools.png(tools.render_grid_section("i", act["il"], "full", act["twt"], (act["xl"], act["twt"]), source="both"))
            return (json.dumps(d, default=str) + (" A picture of the inline through the largest difference is attached: thick lines "
                                                  "labelled at the left are RGT horizons (H#), thin lines labelled at the right are "
                                                  "the second-opinion horizons (G#)." if png else "")), png, act
        if name == "dossier":
            if a.get("lead"):
                d = tools.dossier(lead_id=a["lead"])
            else:
                d = tools.dossier(float(a["il"]), float(a["xl"]), float(a["twt"]))
            return json.dumps(d, default=str), None, (d.get("place") and {"il": d["place"]["il"], "xl": d["place"]["xl"], "twt": d["place"]["twt_ms"]})
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
        "look_slice": f"Looking at the time slice at {a.get('twt')} ms",
        "look_well": f"Looking at the logs of well {a.get('name', '')}",
        "fault": f"Reading what is known about fault {a.get('id', '')}",
        "compare_horizons": f"Comparing the two horizon interpretations {a.get('name', '')}",
        "dossier": f"Collecting all tied evidence for {a.get('lead') or 'this place'}",
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
    """Step 1, 'locate': done by the app, so the model starts from the facts of the place."""
    if not view:
        return ""
    try:
        bits = [f"inline {view['il']}", f"crossline {view['xl']}", f"TWT {view['twt']} ms", f"volume {view.get('vol_label', view.get('vol'))}"]
        p = tools.probe(float(view["il"]), float(view["xl"]), float(view["twt"])) if tools.grid() is not None else {}
        if p.get("no_data_here"):
            bits.append("NO DATA HERE (empty survey edge)")
        if p.get("between_horizons"):
            bits.append(f"between {p['between_horizons'][0]} and {p['between_horizons'][1]}")
        if p.get("inside_leads"):
            bits.append("inside " + ", ".join(p["inside_leads"]))
        if p.get("facies_name"):
            bits.append(f"seismic facies {p['facies']} ({p['facies_name']})")
        sc = p.get("sand_certainty")
        if sc:
            bits.append(f"rock here: {sc['rock']} with certainty {sc['certainty']} ({sc['basis']})")
        if p.get("nearest_well"):
            bits.append(f"nearest well {p['nearest_well']['well']} {p['nearest_well']['distance_km']} km away")
        if p.get("nearest_fault"):
            bits.append(f"nearest named fault {p['nearest_fault']['fault']} {p['nearest_fault']['distance_km']} km away")
        ha = p.get("horizons_agree")
        if ha:
            bits.append(f"the two horizon interpretations ({ha['rgt']} / {ha['assistant']}) differ by {ha['difference_here_ms']} ms here")
        return "\n\n(The user's crosshair is at: " + "; ".join(bits) + ".)"
    except Exception:
        return ""


def _gather(question):
    """Step 2, 'gather': anything the question names (a lead, fault, well, horizon) is looked up before the
    model is asked, so it does not spend a tool call on it. Returns (text to append, step lines)."""
    k = knowledge.load() or {}
    out, steps = [], []
    q = question or ""
    for lid in list(dict.fromkeys(re.findall(r"\bL\d+\b", q)))[:2]:
        d = tools.dossier(lead_id=lid)
        if "error" not in d:
            out.append(f"LEAD {lid}: " + json.dumps(d, default=str)[:3500])
            steps.append(f"Gathered all tied evidence for {lid}")
    for fid in list(dict.fromkeys(re.findall(r"\bF\d+\b", q)))[:2]:
        d = tools.fault(fid)
        if "error" not in d:
            out.append(f"FAULT {fid}: " + json.dumps(d, default=str)[:1500])
            steps.append(f"Gathered what is known about fault {fid}")
    for gid in list(dict.fromkeys(re.findall(r"\bG\d+\b", q)))[:2]:
        d = tools.compare_horizons(gid)
        if "error" not in d:
            out.append(f"HORIZON {gid}: " + json.dumps(d, default=str)[:1500])
            steps.append(f"Gathered the comparison for horizon {gid}")
    low = q.lower()
    n = 0
    for w in k.get("wells", []):
        if n < 2 and re.search(r"(?<![a-z0-9])" + re.escape(w["name"].lower()) + r"(?![a-z0-9])", low):
            d = tools.well(w["name"])
            keep = {a: d.get(a) for a in ("name", "il", "xl", "td_source", "zones", "assistant_reading", "info", "notes", "horizons")}
            out.append(f"WELL {w['name']}: " + json.dumps(keep, default=str)[:3000])
            steps.append(f"Gathered the log zones and horizon crossings of well {w['name']}")
            n += 1
    if not out:
        return "", []
    return "\n\n(Looked up by the app for this question:\n" + "\n".join(out) + ")", steps


def _prune_images(messages, keep=None):
    """Keep only the newest pictures in the conversation; older ones are replaced by a short note.

    How many the server accepts in one request is measured by scripts/gemma_probe.py (default 2)."""
    keep = int(caps().get("max_images", 2)) if keep is None else keep
    seen = 0
    for m in reversed(messages):
        if not isinstance(m.get("content"), list):
            continue
        for part in reversed(m["content"]):
            if part.get("type") == "image_url":
                seen += 1
                if seen > keep:
                    part.clear()
                    part.update({"type": "text", "text": "(an earlier picture was removed here to save space)"})
    return messages


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
    note = _view_note(view)
    if note:
        short = re.sub(r" \((?:checked on|sand-probability volume)[^)]*\)", "", note.strip().strip("()").replace("The user's crosshair is at: ", ""))
        yield {"type": "step", "text": "Located: " + short.rstrip(".")[:320]}
    extra, gsteps = _gather(last["content"])
    for s in gsteps:
        yield {"type": "step", "text": s}
    content = [{"type": "text", "text": last["content"] + note + extra}]
    if attach_view and view:
        try:
            content.append(_img(tools.render_view(view)))
            content[0]["text"] += ("\n(One picture of what the user is looking at is attached. LEFT: the vertical section, with "
                                   "horizons H# labelled at the left edge, second-opinion horizons G# at the right edge, fault "
                                   "sticks in red. RIGHT: the time slice at the crosshair time (map view) with wells and named "
                                   "faults. The cyan circle labelled 'you' marks the crosshair in both.)")
            yield {"type": "step", "text": "Looking at your section and time slice"}
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
            for piece in _stream(_prune_images(messages)):
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


# ====================================================================== structured questions (the AI pass)
# The interpretation pass asks Gemma many small questions about pictures. Each answer must be JSON that
# refers to numbered marks drawn by the app, never free coordinates. Answers are cached on disk.
STATS = {"asked": 0, "cached": 0, "failed": 0, "seconds": 0.0}
_budget = {"deadline": None}
_json_mode = {"ok": None}
_lock = threading.Lock()


def set_budget(minutes):
    """After this many minutes ask_json stops asking and returns None (callers fall back to their own choice)."""
    _budget["deadline"] = None if not minutes else time.time() + minutes * 60.0


def budget_left():
    return _budget["deadline"] is None or time.time() < _budget["deadline"]


def parse_json(text):
    """Tolerant JSON reader: drops thinking blocks and code fences, finds the outermost object or list."""
    t = _visible(text or "").strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    if not starts:
        return None
    a = min(starts)
    b = max(t.rfind("}"), t.rfind("]"))
    if b <= a:
        return None
    body = t[a:b + 1]
    for cand in (body, re.sub(r",\s*([}\]])", r"\1", body), re.sub(r",\s*([}\]])", r"\1", body).replace("'", '"')):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def caps():
    """What the probe script measured about the shared model (empty if it was never run)."""
    return config.read_json(config.PROBE_FILE, {}) or {}


def _post_json(messages, max_tokens, temperature):
    url = config.LLM_URL.rstrip("/") + "/chat/completions"
    body = {"model": config.LLM_MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if _json_mode["ok"] is not False:
        r = requests.post(url, timeout=(10, 240), json=dict(body, response_format={"type": "json_object"}))
        if r.status_code == 200:
            _json_mode["ok"] = True
            return r.json()["choices"][0]["message"].get("content") or ""
        _json_mode["ok"] = False                    # this server does not offer JSON mode: plain text from now on
    r = requests.post(url, timeout=(10, 240), json=body)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"].get("content") or ""


def ask_json(task, prompt, image=None, mock=None, max_tokens=500, temperature=0.1, use_cache=True):
    """One question, one optional picture, one JSON answer (dict or list) - or None if it could not be had.

    task  : short name, used for the cache file and the log
    mock  : the answer the stand-in model gives (the caller's own best choice), used when COPILOT_LLM_URL=mock
    """
    key = hashlib.sha1((config.LLM_MODEL + "|" + task + "|" + prompt).encode() + (image or b"")).hexdigest()[:20]
    path = config.LLM_CACHE / f"{re.sub(r'[^a-z0-9_]+', '_', task.lower())}_{key}.json"
    if use_cache:
        hit = config.read_json(path)
        if hit is not None and "answer" in hit:
            with _lock:
                STATS["cached"] += 1
            return hit["answer"]
    if not budget_left():
        return None
    t0 = time.time()
    ans, raw = None, ""
    try:
        if config.LLM_URL == "mock":
            mode = os.environ.get("COPILOT_MOCK_MODE", "")
            if mode == "timeout":
                raise TimeoutError("stand-in model: timeout")
            if mode == "garbage":
                raw = "I think the answer is probably yes, but {not json"
            elif mode == "wrongids":
                raw = json.dumps(_wrong_ids(mock))
            else:
                raw = json.dumps(mock)
            ans = parse_json(raw)
        else:
            content = [{"type": "text", "text": prompt}]
            if image:
                content.append(_img(image))
            messages = [{"role": "user", "content": content if image else prompt}]
            for attempt in range(2):
                raw = _post_json(messages, max_tokens, temperature)
                ans = parse_json(raw)
                if ans is not None:
                    break
                messages = messages[:1] + [{"role": "assistant", "content": raw[:800]},
                                           {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON, nothing else."}]
    except Exception as e:
        raw = f"{type(e).__name__}: {e}"
        ans = None
    with _lock:
        STATS["asked"] += 1
        STATS["seconds"] += time.time() - t0
        if ans is None:
            STATS["failed"] += 1
    if ans is not None and use_cache:
        try:
            config.write_json(path, {"task": task, "answer": ans, "seconds": round(time.time() - t0, 1), "question": prompt})
            if image:                               # keep the picture too: it shows exactly what the model was asked about
                path.with_suffix(".png").write_bytes(image)
        except Exception:
            pass
    return ans


def _wrong_ids(obj):
    """Test mode: every id in the stand-in answer is replaced by one that does not exist."""
    if isinstance(obj, dict):
        return {k: (9999 if k == "id" else _wrong_ids(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_wrong_ids(v) for v in obj]
    return obj


def ask_many(jobs, workers=None):
    """Run several ask_json calls side by side (a few at a time: the model is shared). jobs: list of kwargs."""
    if not jobs:
        return []
    n = max(1, min(int(workers or config.AI_WORKERS), int(caps().get("workers", 99)), len(jobs)))
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(lambda kw: ask_json(**kw), jobs))
