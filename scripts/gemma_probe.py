"""One-minute check of the shared Gemma model, to run on the HPC before the first interpretation.

    .venv/bin/python scripts/gemma_probe.py

It asks the model a few questions whose answers are known: can it be reached, does it accept pictures, can it
read the numbered marks and lettered grid squares the app draws, and how long does a question take. The result
is printed in plain words and saved to workspace/gemma_probe.json, where the app reads it.
Only made-up test pictures are sent: no survey data.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from copilot import config, llm, tools  # noqa: E402

RULE = "Reply with JSON only, no other text.\n"


def layers(seed, n=420, dip=0.0, bump=0.0):
    """A made-up layered section (rows = time): wavy black and white bands, like seismic."""
    rng = np.random.default_rng(seed)
    x = np.arange(n)[None, :]
    t = np.arange(n)[:, None]
    shift = dip * x + bump * np.exp(-((x - n / 2) / (n / 5.0)) ** 2)
    ph = rng.uniform(0, 6.28, 5)
    fr = rng.uniform(0.05, 0.16, 5)
    s = sum(np.sin(fr[k] * (t - shift) + ph[k]) for k in range(5)) + 0.15 * rng.standard_normal((n, n))
    return s.astype(np.float32), shift[0]


def picture(s, lines=(), marks=(), cells=None, title=""):
    ext = {"h": [0.0, float(s.shape[1] - 1)], "v": [0.0, float(s.shape[0] - 1)]}
    return tools.png(tools.render_panel(tools.to_rgb(s, -3.5, 3.5, "gray"), ext, title=title, lines=lines, marks=marks, cells=cells))


def ask(prompt, image=None, tokens=120):
    t0 = time.time()
    a = llm.ask_json("probe", RULE + prompt, image, mock={"ok": True}, max_tokens=tokens, use_cache=False)
    return a, time.time() - t0


def main():
    out = {"when": time.strftime("%Y-%m-%d %H:%M"), "url": config.LLM_URL, "model": config.LLM_MODEL}
    print(f"\nChecking the assistant at {config.LLM_URL}\n")
    ok, msg = llm.check()
    out["reachable"] = bool(ok)
    print(f"  [{'OK' if ok else '--'}]  The model answers ({msg})")
    if not ok:
        print("\n  The model cannot be reached from here. The app still works, but the assistant and its own pass do not.")
        config.write_json(config.PROBE_FILE, out)
        return
    if config.LLM_URL == "mock":
        print("  (This is the built-in stand-in, not Gemma: the checks below are skipped.)")
        config.write_json(config.PROBE_FILE, out)
        return
    try:
        import requests
        r = requests.get(config.LLM_URL.rstrip("/") + "/models", timeout=10).json()
        m = next((q for q in r.get("data", []) if q.get("id") == config.LLM_MODEL), (r.get("data") or [{}])[0])
        if m.get("max_model_len"):
            out["max_model_len"] = int(m["max_model_len"])
            print(f"  [OK]  It can read up to {out['max_model_len']} tokens per question")
    except Exception:
        pass

    a, sec = ask('What is 17 + 25? JSON: {"sum": <number>}')
    out["text_ok"], out["seconds_text"] = bool(isinstance(a, dict) and a.get("sum") == 42), round(sec, 1)
    out["json_mode"] = llm._json_mode["ok"]
    print(f"  [{'OK' if out['text_ok'] else '--'}]  A plain question is answered in JSON ({sec:.1f} s)")

    # 1. numbered marks: which number sits on the red square?
    hits, secs, n = 0, [], 4
    for k in range(n):
        s, _ = layers(k)
        rng = np.random.default_rng(100 + k)
        pts = [(float(rng.uniform(40, 380)), float(rng.uniform(40, 380))) for _ in range(6)]
        target = int(rng.integers(6))
        rgb = tools.to_rgb(s, -3.5, 3.5, "gray")
        x, y = int(pts[target][0]), int(pts[target][1])
        rgb[max(0, y - 14):y + 14, max(0, x - 14):x + 14] = (230, 30, 30)
        ext = {"h": [0.0, 419.0], "v": [0.0, 419.0]}
        img = tools.png(tools.render_panel(rgb, ext, title="Test picture", marks=[{"h": p[0], "v": p[1], "label": str(i + 1)} for i, p in enumerate(pts)]))
        a, sec = ask('Six numbered circles are drawn on the picture. One of them sits on a red square. Which number? JSON: {"id": <number>}', img)
        secs.append(sec)
        hits += bool(isinstance(a, dict) and str(a.get("id")) == str(target + 1))
        if k == 0:
            out["images_ok"] = a is not None
            if a is None:
                break
    out["mark_accuracy"] = round(hits / n, 2)
    out["seconds_image"] = round(float(np.mean(secs)), 1)
    print(f"  [{'OK' if out.get('images_ok') else '--'}]  It accepts pictures ({out['seconds_image']} s per picture question)")
    if not out.get("images_ok"):
        print("\n  The model did not answer a question with a picture. Its own pass cannot use pictures; tell the mentors\n"
              "  and send them this output. The app still answers from the measured evidence.")
        config.write_json(config.PROBE_FILE, out)
        return
    print(f"  [{'OK' if hits >= 3 else '--'}]  It reads numbered marks: {hits} of {n} right")

    # 2. which line does NOT follow a reflector? (the same kind of question as the app's decoys)
    hits, n = 0, 4
    for k in range(n):
        s, shift = layers(20 + k, dip=0.08 * (-1) ** k, bump=28.0)
        rng = np.random.default_rng(200 + k)
        bases = sorted(rng.choice(np.arange(60, 360, 45), size=4, replace=False).tolist())
        wrong = int(rng.integers(4))
        lines = []
        for i, b in enumerate(bases):
            z = b + shift + (np.linspace(-34, 34, 420) if i == wrong else 0)
            lines.append({"pts": list(zip(np.arange(420.0).tolist(), np.clip(z, 2, 417).tolist())), "label": str(i + 1), "at": 0.15 + 0.2 * i})
        a, _ = ask("Four numbered coloured lines are drawn on a layered section. Three follow the layering; one cuts across "
                   'the layers. Which number cuts across? JSON: {"id": <number>}', picture(s, lines=lines, title="Test picture"))
        hits += bool(isinstance(a, dict) and str(a.get("id")) == str(wrong + 1))
    out["line_accuracy"] = round(hits / n, 2)
    print(f"  [{'OK' if hits >= 3 else '--'}]  It spots a line that cuts across the layers: {hits} of {n} right")

    # 3. lettered grid squares: where is the bright patch?
    hits, near, n = 0, 0, 4
    for k in range(n):
        rng = np.random.default_rng(300 + k)
        s = 0.5 * rng.standard_normal((420, 420)).astype(np.float32)
        col, row = int(rng.integers(8)), int(rng.integers(8))
        cy, cx = int((row + 0.5) * 52.5), int((col + 0.5) * 52.5)
        yy, xx = np.mgrid[0:420, 0:420]
        s += 6.0 * np.exp(-(((yy - cy) / 13.0) ** 2 + ((xx - cx) / 13.0) ** 2))
        a, _ = ask('The picture has a grid: columns A-H, rows 1-8. One square holds a bright white patch. Which square? JSON: {"cell": "C4"}',
                   picture(s, cells=(8, 8), title="Test picture"))
        got = str((a or {}).get("cell", "")).upper().strip() if isinstance(a, dict) else ""
        hits += got == tools.cell_name(col, row)
        if len(got) == 2 and got[1].isdigit():
            near += abs(ord(got[0]) - 65 - col) <= 1 and abs(int(got[1]) - 1 - row) <= 1
    out["cell_accuracy"], out["cell_accuracy_within_one"] = round(hits / n, 2), round(near / n, 2)
    print(f"  [{'OK' if near >= 3 else '--'}]  It names grid squares: {hits} of {n} exactly right, {near} of {n} within one square")

    # 3b. two pictures in one question (the chat can show the model a second picture after the first)
    try:
        import requests
        tiny = picture(layers(1)[0][:120, :120], title="A")
        body = {"model": config.LLM_MODEL, "max_tokens": 20, "temperature": 0.0,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "How many pictures do you see? Answer with a number."},
                                                         llm._img(tiny), llm._img(tiny)]}]}
        r = requests.post(config.LLM_URL.rstrip("/") + "/chat/completions", json=body, timeout=(10, 120))
        out["max_images"] = 2 if r.status_code == 200 else 1
    except Exception:
        out["max_images"] = 1
    print(f"  [OK]  Pictures per question the server accepts: {'two or more' if out['max_images'] > 1 else 'one (the chat will keep only the newest picture)'}")

    # 4. three questions at once (the app sends at most three at a time)
    t0 = time.time()
    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(lambda q: ask(f'What is {q} + 1? JSON: {{"sum": <number>}}')[0], (1, 2, 3)))
    par = time.time() - t0
    good = sum(isinstance(r, dict) for r in res)
    out["workers"] = 3 if good == 3 and par < 2.5 * max(out["seconds_text"], 0.5) + 3 else 1
    print(f"  [{'OK' if good == 3 else '--'}]  Three questions at the same time: {good} answered in {par:.1f} s -> the app will send "
          f"{out['workers']} at a time")

    est = round(130 * out["seconds_image"] / out["workers"] / 60.0, 0)
    out["estimated_pass_minutes"] = est
    print(f"\n  The assistant's own pass asks about 130 picture questions: roughly {est:.0f} minutes here "
          f"(it stops by itself after {config.AI_MINUTES:.0f} minutes and keeps what it has).")
    weak = [w for w, v in (("numbered marks", out["mark_accuracy"]), ("lines across layers", out["line_accuracy"]),
                           ("grid squares", out["cell_accuracy_within_one"])) if v < 0.75]
    if weak:
        print("  Weak spots: " + ", ".join(weak) + ". The app checks the assistant with deliberately wrong marks and ignores its\n"
              "  votes when it fails that check, so the results stay safe; they are then the app's own choices.")
    else:
        print("  The model reads the app's pictures well enough for its own pass.")
    config.write_json(config.PROBE_FILE, out)
    print(f"\n  Saved to {config.PROBE_FILE}\n")


if __name__ == "__main__":
    main()
