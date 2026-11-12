# Exploration Copilot

An AI assistant that interprets a 3-D seismic survey with its wells and then **reasons about it like a
geoscientist**: it states what the data shows, what that most likely means, what else it could mean, how
sure it is, and what to do next — and every statement links back to the evidence and to the place in the
data where you can check it.

Built for the APGCE 2026 GeoHackathon, Challenge 1. It runs entirely on the hackathon HPC; no data leaves it.

## What it does

| Step | What happens | Built on |
|---|---|---|
| 1. Load | SEG-Y volumes become local OpenVDS files with lossless Zip compression and full-resolution reads. Geometry is checked before a store is published. Wells (logs, deviation surveys, checkshots, well list) are their own input and are placed in the seismic. | [OpenVDS](https://pypi.org/project/openvds/), segyio, lasio |
| 2. View | Inline / crossline sections and time slices with linked crosshairs, zoom buttons, every volume including a coherence cube, a well-log panel, overlays for both sets of horizons, fault sticks, facies, wells and leads. | plain JavaScript canvas |
| 3. Interpret | Relative Geologic Time for the whole cube, horizons as iso-RGT surfaces, coherence and fault likelihood, seismic facies from foundation-model features with standard facies names, spill-point trap analysis, sand bodies. Empty traces at the survey edge are found and left out. | [RGT-Est](https://github.com/douyimin/RGT-Est), [NCS models](https://github.com/NorskRegnesentral/NCS_models) |
| 4. Tie | Each well is tied to the seismic (best time shift), horizons are compared with log boundaries, and the sand prediction is tested on the wells, each well predicted from the others, so a rock call comes with an honest certainty. | this repository |
| 5. Second opinion | Gemma looks at pictures and works like a second interpreter: it chooses the key reflectors, carries them across faults, accepts or rejects fault sticks, reads time slices and well logs, and reviews the facies names. It chooses between numbered marks; the app draws and measures. The result is compared with RGT. | Gemma 4 via vLLM |
| 6. Rank | Each trap becomes a *lead* with transparent scores for trap, reservoir, seal and hydrocarbon indicators, and a list of numbered facts for, against and unknown, drawn from all of the above. | this repository |
| 7. Ask | The chat assistant answers from those facts, can look things up, and can **look at pictures** of sections, time slices, maps and logs. It cites facts like `[E12]`; clicking one jumps the viewer there. | Gemma 4 via vLLM |

## Why the answers can be trusted (and checked)

* **The language model never guesses numbers.** It cannot read a seismic cube, so the pipeline measures
  everything first and hands it a briefing in which every fact has an id and a location.
* **A fixed way of arguing.** Observation → Interpretation → Alternative → Confidence → Recommendation.
* **Evidence you can click.** `[E7]` shows the fact and moves the crosshair to it. "Show the facts used"
  lists everything an answer relied on. "Challenge this" makes the assistant argue against itself.
* **The assistant can look.** It can request a picture of any section or map (with horizons, leads and
  wells drawn on it) and reason from what it sees, like an interpreter at a workstation.
* **Three kinds of evidence, kept apart.** Measurements from the seismic (blue tags), what the wells show
  (green), and the assistant's own readings of pictures (purple). Each visual reading carries a measured check.
* **Two interpreters.** Horizons come from RGT-Est (H1, H2 ...) and, independently, from reflectors chosen on a
  picture and tracked along a net of lines (G1, G2 ...). Where they agree the structure is robust; where they
  differ, the difference map shows it and the lead's evidence says so.
* **The assistant is tested, not trusted.** Every picture it judges also holds a mark that is wrong on purpose
  (a fault stick where nothing is faulted, a line drawn across the reflectors). How many of these it rejects is
  reported, and if it accepts them its votes are ignored.
* **Certainty that means something.** "Sand, 0.87" is what the wells showed when the prediction was that high,
  with the number of test intervals and wells beside it.
* **Honest limits.** Borrowed checkshots, fallback methods, thinned grids, empty survey edges and anything the
  assistant did not review are listed and passed to the assistant, which is told to mention them.
* **Wells keep the seismic honest.** Sand probability and impedance are compared with the logs at every
  well/horizon crossing, and the result is part of the evidence.

## How a lead is scored

Simple on purpose, so a geoscientist can disagree with any part of it:

* **Trap** — closure relief and area from spill-point analysis of the horizon time map (four-way closures;
  fault-dependent closures found by repeating the analysis with faults treated as seals; sand bodies that
  ignore structure are flagged as stratigraphic candidates, with an up-dip pinch-out test).
* **Reservoir** — sand probability in the sandiest zone of the layer under the trap.
* **Seal** — how shale-prone the zone directly above is.
* **Hydrocarbon indicators** — low P-impedance, low Vp/Vs, far-versus-near amplitude increase, and whether
  the bright amplitude fits the closure.
* Wells that already drilled the lead are reported with what their logs show in that zone.

Scores are screening aids, not probabilities of success; the assistant says so.

## Robust by design

Every heavy component has a fallback, so the app always produces a result and says which path it took:

| Component | First choice | Fallback |
|---|---|---|
| SEG-Y conversion | OpenVDS → local lossless VDS | segyio → NumPy store |
| Relative geologic time | RGT-Est on a GPU through Slurm | RGT-Est on CPU, then a classical dip-steered RGT |
| Seismic facies | NCS 2.5D foundation model | NCS 2D, then Sheng et al. SFM (if its checkpoint is supplied), then classical attributes |
| Assistant | Gemma on the HPC | clear message that it is offline; viewer and leads still work |
| Assistant's own pass | Gemma chooses between numbered marks on pictures | the app's own best match, labelled "not reviewed" |
| Rock certainty | several volumes combined, if that tests better | the sand-probability volume re-scaled to the wells, or as supplied |

## Run it

```bash
git clone https://github.com/AtifAsrul/exploration-copilot.git
cd exploration-copilot
bash setup.sh     # once: environments, packages, models
bash start.sh     # every time: then open http://127.0.0.1:8050
```

Run these commands in your writable project folder on the HPC. Open the browser inside the HPC remote
desktop; use the address printed by the server. For an existing checkout, follow
[Updating to this version](HPC_GUIDE.md#updating-to-this-version) instead of cloning again.

In the page: **Import data** → **Run interpretation** → ask. A plain-language walkthrough is in
[HPC_GUIDE.md](HPC_GUIDE.md).

Without the page: `.venv/bin/python -m copilot.pipeline --data /path/to/data`.

Setup/start now show terminal stage bars, elapsed-time heartbeats and local log output.
Size reports show model/store/interpretation/log sizes and measured net growth every 30 seconds.
OpenVDS import logs report input/output sizes, elapsed time and end-to-end source MB/s.
To watch an existing run without restarting it: `.venv/bin/python -u scripts/terminal.py watch`.
The monitor reads only this app's status and logs; it does not change Slurm jobs or system settings.
See [Watching progress and debugging](HPC_GUIDE.md#watching-progress-and-debugging) for log paths
and update commands. Stage bars describe step order, not estimated time remaining.

Try it without real data: `python tests/make_synthetic.py demo_data --dead --wells 6`, then import `demo_data`.
Check the new parts against that survey's known answer: `python -m unittest tests.test_v4`.

## Layout

```
copilot/store.py       SEG-Y -> AI-ready store, fast slicing
copilot/wells.py       well list, deviations, checkshots, LAS logs
copilot/rgt.py         RGT-Est, classical RGT, horizons from RGT
copilot/faults.py      coherence, fault likelihood and fault inventory
copilot/calibrate.py   well ties, horizons against log boundaries, rock certainty checked on the wells
copilot/tracker.py     follows reflectors along a net of lines and fills in between
copilot/sticks.py      fault sticks, throw across them, joining sticks into named faults
copilot/ai_pass.py     the assistant's own pass: pictures with numbered marks, questions, decoy checks
copilot/sfm.py         foundation-model features, facies, look-alike search
copilot/attributes.py  maps, traps, leads, scores, evidence
copilot/knowledge.py   the evidence file and briefing the assistant reads
copilot/tools.py       look-ups and pictures for the page and the assistant
copilot/llm.py         the assistant: rules, tools, Gemma client
copilot/pipeline.py    runs all steps; GPU steps through Slurm
scripts/gemma_probe.py one-minute check that Gemma reads the app's pictures
copilot/server.py      web server (this machine only)
web/                   the page
```

## Security and privacy

The server listens on `127.0.0.1` only. The only network calls are to the Gemma endpoint on the same
HPC. Seismic and well data stay where they are.

## Credits and licences

* OpenVDS — The Open Group / Bluware. The app uses the pinned PyPI binary distribution;
  see [package licensing](https://pypi.org/project/openvds/) for the terms of that distribution.
* Existing SDIP/MDIO stores remain readable for compatibility; SDIP is no longer installed or invoked.
* RGT-Est — Dou, Wu, Gao, Bi (2026), *Learning Stratigraphically Consistent Relative Geologic Time from 3D
  Seismic Data via Sinusoidal Mapping*; code MIT, weights CC BY 4.0. The iso-surface extraction follows the
  helper shipped with RGT-Est (Xinming Wu).
* NCS models — Ordoñez et al. (2025), *The NCS-model*, Norsk Regnesentral, Apache-2.0.
* Seismic Foundation Model — Sheng et al. (2023), optional.
* Gemma — Google.
