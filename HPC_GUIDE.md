# Getting Exploration Copilot running on the HPC

Follow the parts in order. Type each command exactly, one line at a time, and press Enter after each.
Tip: you can copy a line here and paste it into the HPC terminal with **Ctrl + Shift + V**.

**Already have an older copy on the HPC?** Go straight to
[Updating to this version](#updating-to-this-version), then come back to Part D.

---

## Part A — Repository and prerequisites

Use [AtifAsrul/exploration-copilot](https://github.com/AtifAsrul/exploration-copilot), branch `main`.
The repository contains code; challenge data, models, environments and interpretation results stay on the HPC.

Before installing, check `python3 --version`, `git --version`, and `df -h .` in your writable HPC
project folder. Setup needs internet access for Python packages and roughly 1 GB of model weights;
PyTorch and converted seismic stores need additional space. Installation time depends on the network.
Use the organisers' supported Python environment if the system Python cannot install the requirements.
Do not use `sudo` or change the system Python.

---

## Part B — Install on the HPC (once)

1. Connect with RustDesk and log in, as in the organisers' manual.
2. Open a terminal: top-left menu **Applications → System Tools → Terminal**.
3. Type these four lines. If the folder already exists, use the update section instead:

```
cd /data/epic20/APGCE/$USER
git clone https://github.com/AtifAsrul/exploration-copilot.git
cd exploration-copilot
bash setup.sh
```

4. Wait. Text scrolls by while it installs and downloads the models. It ends with a list called
   **Self-check**. Read each `[--]`: it identifies a missing component or unavailable service. Some
   components have fallbacks; an unavailable Gemma service means chat is offline. A zero exit code
   from the self-check does not mean every component passed. Required package failures stop setup.

5. Check the assistant's image support with `.venv/bin/python scripts/gemma_probe.py`.
   This uses made-up test pictures. Review any `[--]` results before relying on visual interpretations.

If a line fails:

| What you see | What to type instead |
|---|---|
| `cd: /data/epic20/APGCE/...: No such file or directory` | `ls /data/epic20/APGCE/` to see your team's folder name, then `cd /data/epic20/APGCE/that_name` |
| `git: command not found` | `curl -L -o ec.zip https://github.com/AtifAsrul/exploration-copilot/archive/refs/heads/main.zip` then `unzip ec.zip` then `cd exploration-copilot-main` then `bash setup.sh`. This ZIP copy cannot use `git pull`; use a fresh ZIP in a separate folder for later updates. |
| GitHub cannot be reached at all | In RustDesk use the **file transfer** button to copy `exploration-copilot-v4.zip` to the HPC, then `unzip exploration-copilot-v4.zip`, `cd exploration-copilot`, `bash setup.sh` |

---

## Part C — Start the app (every time)

In the terminal, inside the `exploration-copilot` folder:

```
bash start.sh
```

Firefox opens by itself. If it does not, open Firefox and type `http://127.0.0.1:8050` in the address bar.
Keep the terminal window open while you use the app.
Use Firefox **inside the HPC remote desktop**. If 8050 is busy, use the address printed by the server.
For an SSH-only connection, forward the printed port from your laptop, for example
`ssh -N -L 8050:127.0.0.1:8050 YOUR_HPC_USER@YOUR_HPC_HOST`, using your approved access route,
then open `http://127.0.0.1:8050` on your laptop. Substitute the printed port if it differs.

If you previously used a custom workspace, set it in this terminal before checks and startup:
`export COPILOT_WORKSPACE="/absolute/path/to/your/existing/workspace"`.
Use that same value for setup, the image probe, the pipeline and the monitor. The default is
`workspace/` inside the checkout. Keep an existing workspace when you want to reuse its results.

### Updating to this version

Routine code updates reuse your saved workspace. This well-folder fix requires no new packages and
no seismic reinterpretation. Run `bash setup.sh` only if the environment is missing or incomplete.

1. Let any active interpretation finish, then stop the app with **Ctrl+C** in its terminal.
2. **On the HPC**, in that terminal:

```bash
cd /data/epic20/APGCE/$USER/exploration-copilot
```

```bash
git remote get-url origin
git status --short
```

   The remote should be `https://github.com/AtifAsrul/exploration-copilot.git` (or its SSH equivalent).
   If it points elsewhere, confirm you are in the intended checkout before changing anything.
   If `git status --short` lists local code edits, preserve them with `git stash push -u -m before-v4-update`.
   Ignored environments, models and workspaces remain in place. Then pull:

```bash
git pull --ff-only origin main
git log -1 --oneline
```

   If Git reports divergent history, stop and review the local commits. Do not reset or overwrite them.
   A saved stash can be reviewed later with `git stash show -p`; reapply it only after reviewing compatibility.

3. For interface or import-selection changes, start the app with `bash start.sh` and refresh Firefox.
   Keep the same `COPILOT_WORKSPACE` value as the existing interpretation. `git pull` and startup
   preserve converted seismic, RGT, facies, horizons and other saved results.

**Repair the previously selected seismic-only well folder.** With the app stopped, use:

```bash
.venv/bin/python -u -m copilot.pipeline \
  --wells /data/epic20/APGCE/input_data/challenge1_challenge2/well_data \
  --force --stages wells,calibrate,leads,knowledge,dossier
bash start.sh
```

This saves the correct well-folder selection and rebuilds wells, well ties/calibration, leads and evidence.
It keeps converted seismic, the interpretation grid, RGT, facies, horizons and fault/coherence cubes.
Use your actual well folder if it differs. The import dialog now finds a neighbouring `well_data/`
folder when browsing `stack/`, and rejects a selected well folder containing no recognised well files.
An invalid saved well path or reader failure stops the well stage with an error, rather than reporting
successful completion with zero wells. The well counter refreshes after the well stage.

For later well-data changes, **Import data → Update wells only** performs the same partial refresh.
Avoid **Re-run interpretation** for routine code updates: that action explicitly clears interpretation
outputs. Changes to seismic algorithms, input volumes or grid settings can require recalculating the
affected stages and their dependent results; cached files do not automatically track code changes.

### Watching progress and debugging

`bash setup.sh` now shows a six-step bar, pip download/install output, measured RGT download
progress, and a heartbeat every 15 seconds while a command is running. Each command reports its
elapsed time and exit code. Optional failures are marked `[WARNING]`; a required package install
failure stops setup. The self-check's `[--]` lines still identify unavailable components.

`bash start.sh` also shows interpretation stage position, live pipeline/conversion/GPU log output,
and elapsed-time heartbeats in the same terminal when you click **Run interpretation** in the browser.
Stage bars show order through the steps, **not percentage of total run time**. Some steps take much
longer than others, and a quiet heartbeat confirms the command is still running, not that it is
making measurable progress. The status age helps identify stale updates. Existing fallback methods
and Slurm settings are unchanged.

To attach a read-only monitor in a **second HPC terminal**, without restarting an existing app:

```bash
cd /data/epic20/APGCE/$USER/exploration-copilot
.venv/bin/python -u scripts/terminal.py watch
```

Press **Ctrl+C** in this second terminal to stop only the monitor. It reads this app's existing files;
it does not start/cancel jobs, install anything, or modify data. It starts
following logs from their current ends. If `COPILOT_WORKSPACE` was set for your app, use the same
setting here (or pass `--workspace /your/app/workspace`).

Logs remain inside `workspace/logs/` (or your configured workspace):

* `setup.log`: append-only setup output, including failures.
* `server.log`: append-only server output and terminal monitoring from `bash start.sh`.
* `pipeline.log`: current interpretation output.
* `openvds_*.log`: live OpenVDS importer progress and errors, available before conversion finishes.
* `sdip_*.log`: legacy converter logs, if an older run produced them.
* `gpu_job_*.log`: Slurm worker output, once the GPU job starts.

For plain log output in a second terminal:

```bash
tail -n 60 -F workspace/logs/pipeline.log
```

All monitoring is local to this app, uses Python's standard library, and adds no system services,
administrator commands, or external reporting. Logs may contain local paths and diagnostics;
review them before sharing them.

### Pulling this monitoring update

Inside your existing HPC checkout (do not run the installer again just to get monitoring):

```bash
cd /data/epic20/APGCE/$USER/exploration-copilot
git pull --ff-only origin main
```

For an app that is already running, use the second-terminal monitor above immediately. It leaves
the existing run alone. The updated start script and live converter logging take effect on the next
app/pipeline launch. If setup is not yet complete, run `bash setup.sh` after pulling; otherwise use
`bash start.sh` the next time you launch the app.

### Sizes and speed

Setup, start and the read-only monitor report sizes for `models/` and the configured workspace's
`stores/`, `interp/` and `logs/` every 30 seconds. Reports use decimal MB/GB and show net file-size
growth in MB/s after two samples. Where the filesystem exposes allocated blocks, allocation and
its growth rate are shown separately. Preallocated/sparse files can have a large logical size before
data is written. Net growth is **not disk bandwidth, download speed, or completion percentage**;
CPU/GPU processing can be active while file sizes stay unchanged. Replacement/deletion can cause
negative net growth. Model downloads retain their own byte-based download progress.

The size sampler reads file metadata only, skips symlinks/junctions, and never scans the input-data
folder or the system package/cache directories. Each folder scan is capped at 4,000 entries and a
0.15-second cooperative budget (a filesystem call may itself take longer). A truncated/unreadable
scan is labelled `>=` / `partial scan`; its rate is withheld. To reduce metadata work further:

```bash
.venv/bin/python -u scripts/terminal.py watch --size-interval 60
```

Or omit size scans with `watch --no-sizes`. Directory rates begin when the monitor attaches;
they do not reconstruct earlier throughput. A size report may show zero bytes before an output
folder has been created. Environment installations and pip caches are excluded from the totals.

### Trying OpenVDS instead of SDIP

New conversions now use OpenVDS 3.4.9's `SEGYImport` and local `.vds` files with **lossless Zip**
compression. The viewer and interpretation pipeline read full-resolution data directly through
OpenVDS. No cloud storage or system installation is needed. The importer log includes its own
progress; completion reports SEG-Y size, VDS size, elapsed time and average source MB/s for import
plus validation. This average is a conversion measurement, not raw filesystem bandwidth.

To update an existing HPC installation without rerunning every setup step:

```bash
cd /data/epic20/APGCE/$USER/exploration-copilot
git pull --ff-only origin main
.venv/bin/python -m pip install --only-binary=:all: "openvds==3.4.9"
```

Existing converted stores are reused automatically, including legacy MDIO/NumPy stores. For a
fresh OpenVDS trial that preserves the current interpretation and converted stores, launch with a
separate workspace (this needs additional disk space for new outputs):

```bash
COPILOT_WORKSPACE="$PWD/workspace-openvds" bash start.sh
```

Then import the normal challenge input folder in the browser and run interpretation. The second
terminal monitor must use the same workspace:

```bash
COPILOT_WORKSPACE="$PWD/workspace-openvds" .venv/bin/python -u scripts/terminal.py watch
```

The app validates inline/crossline/sample coordinates and decodes a trace before publishing a
VDS. Irregular spacing or an importer geometry mismatch uses the built-in segyio/NumPy fallback
and explains the reason in the log. In synthetic testing, OpenVDS 3.4.9 inferred incorrect axis
steps for one crossline-sorted fixture; the validation rejected that output and the fallback
preserved the values. The HPC files must still be checked there. `COPILOT_NO_OPENVDS=1` explicitly
selects the fallback for **new** conversions. Existing SDIP environments and original SEG-Y files
are left in place. Stop/relaunch this trial app before expecting a running process to use new code;
the separate read-only monitor can attach without stopping an existing run.

### Recovering a stage-5 OpenBLAS crash

If a GPU log says `OpenBLAS ... too many memory regions`, then `Segmentation fault`, that
worker has crashed. A following Slurm `SPECIAL_EXIT` is a held job, not a running calculation.
Older app versions continued waiting on this state and refreshed their heartbeat indefinitely.

The app now sets process-local BLAS thread limits before loading numerical libraries, uses a
scoped BLAS limit around PCA/clustering, and respects `SLURM_CPUS_PER_TASK` for OpenMP. The
worker script also exports limits before Python starts. These changes do not modify system
Python, rebuild OpenBLAS, or change the HPC's configuration/GPU allocation. Failed/held Slurm
states are detected promptly; the app cancels only the job it submitted before taking its
fallback path. A failure to cancel stops the pipeline instead of launching a competing fallback.

For a run that completed RGT but crashed during facies grouping:

1. Press Ctrl+C in the app terminal to stop the waiting foreground run.
2. In `squeue -u "$USER"`, identify the failed/held job from that run and cancel that specific job
   with `scancel JOB_ID`. Do not cancel another teammate's active job.
3. Pull the update, then resume the remaining stages in the **same workspace**, without `--force`:

```bash
cd /data/epic20/APGCE/$USER/exploration-copilot
git pull --ff-only origin main
COPILOT_WORKSPACE="$PWD/workspace-openvds" .venv/bin/python -u -m copilot.pipeline --stages sfm,horizons,faults,calibrate,leads,knowledge,ai,dossier
```

This preserves converted volumes, the interpretation grid, and `rgt.npy`. Foundation-model
features may be calculated again because the old run did not save a completed facies result.
After the command finishes, launch the app again with the same `COPILOT_WORKSPACE` setting.
Avoid **Re-run interpretation** for recovery: it clears interpretation outputs and starts over.

Check input coverage too: five stack volumes and zero wells indicate a partial-data run. For the
complete challenge, select `/data/epic20/APGCE/input_data/challenge1_challenge2`, which includes
`stack/`, `inversion/`, and `well_data/`. Resuming stages alone does not import missing data or
rebuild a previously saved grid with additional inversion products.

---

## Part D — First run (once — do this well before the judging)

1. Click **Import data** (top right). Two folders are already filled in: line 1 is the seismic, line 2 is
   the well data (logs, deviation surveys, checkshots, well list). Under line 2 the app says how many well
   files it found. Click **Use these folders**.
2. Click **Run interpretation**. A progress bar shows twelve steps. The first step (converting the nine
   volumes) is the slowest. You can browse the seismic as soon as the first volume is ready.
3. Before you click Run, close any Jupyter notebook that uses the **GPU via SLURM** kernel. Each team gets
   one GPU job at a time, and the app needs it.
4. When the step **Briefing** is done the app is ready to use and tells you so. The last two steps are the
   assistant's own pass over the pictures (**Assistant's own interpretation**, **Final evidence**). They take
   up to about 30 minutes on the shared model and stop by themselves after that. You can use the app meanwhile.
5. When the bar says **Finished**, everything is saved. From now on the app opens ready to use.

Two smaller buttons redo only part of the work, without touching the seismic steps:
**Wells tab → Read the wells again**, and **How it works tab → Run the assistant's pass again** (answers it
already gave are reused, so this is quick).

---

## Part E — Using it

**Left side — the viewer**

* **Inline / Crossline** tabs switch the section. **Time slice / Horizon map** tabs switch the map.
* Click anywhere to move the crosshair; both pictures follow.
* Scroll the mouse wheel to step through lines (or through time on the map). Arrow keys work too.
* The **+ − ⤢** buttons at the bottom right of each picture zoom in, zoom out and show everything.
  **Ctrl + scroll** zooms too; drag moves a zoomed picture.
* **Volume** picks which volume is shown (full, near, far, impedance, sand probability…). **Coherence** is
  in the same list: dark lines are places where neighbouring traces differ, which is how faults and channel
  edges show up, best seen on a time slice.
* The round buttons switch layers on and off:
  * **Horizons · RGT**: solid lines H1, H2 … from RGT-Est.
  * **Horizons · Gemma**: dashed lines G1, G2 …, the second opinion picked on a net of lines.
  * **Fault sticks**: red sticks on sections and red fault lines on maps, named F1, F2 …
  * **Fault likelihood**, **Facies**, **Wells**, **Leads**.
* The lower-left panel has three tabs: **Time slice**, **Horizon map**, **Well logs**. On the horizon map,
  choose a G horizon and "difference from the RGT horizon" to see where the two interpretations disagree.
  In **Well logs**, pick a well: shale volume, resistivity, porosity, impedance, and a rock column (yellow
  sand, red hydrocarbon flag), against time or depth, with the horizons drawn across. Click in the logs to
  move the crosshair to that depth.
* **Find look-alikes** highlights places whose seismic resembles the crosshair spot.
* **Explain this spot** sends the assistant a picture of what you are looking at.

**Right side — the assistant**

* **Assistant** tab: ask in plain words. Tags like `E7` are facts — hover to read, click to jump there.
  Blue = measured from the seismic, green = from the wells, purple = the assistant's own reading of a picture.
  Yellow tags jump to a location. Under each answer: **Show the facts used** and **Challenge this**.
  The grey lines above an answer show what it did first: where you are, what it looked up, what it looked at.
* **Leads** tab: the ranked traps with their scores. **Show on map**, **Explain**, **Argue both sides**.
* **Wells** tab: every well, which files it has, whether its time-depth is its own checkshot or borrowed,
  how much sand and hydrocarbon flag its logs show. **Show logs**, **Go to well**, **Explain**.
* **How it works** tab: which methods ran, the two horizon interpretations compared, the named faults, the
  facies names, the rock-certainty table, the well checks, and the limits of the result.

---

## Part F — A 10-minute demo for the judges

The judges give 10 minutes including the demo. This order shows every part once.

1. **See the data (1 min).** Scroll through a few inlines, switch to crossline, click to move the crosshair,
   zoom with the + button. Switch **Volume** to Coherence and step through time slices: the dark lines are
   faults.
2. **Two interpreters (2 min).** Turn on **Horizons · Gemma** next to **Horizons · RGT**. Say: "solid lines
   are RGT-Est, dashed lines are reflectors Gemma chose and the app tracked on a net of lines". Open
   **Horizon map**, choose a G horizon, then "difference from the RGT horizon": white is agreement.
3. **Faults (1 min).** Point at a red fault stick, then ask: *Tell me about fault F1.* The answer gives its
   strike, throw and what Gemma saw on the sticks.
4. **Wells tied to seismic (2 min).** Open the **Wells** tab, click **Show logs** on a well. Show the sand
   column and the horizons crossing the logs. Open **How it works** and show the rock-certainty table: "when
   the prediction was this high, the wells showed sand this often".
5. **A lead, with all evidence tied together (2 min).** Open **Leads**, click **Show on map**, then
   **Explain**. Point out the three tag colours and click one of each. Click **Challenge this**.
6. **Look for yourself (1 min).** Put the crosshair somewhere interesting, tick **Show the assistant my
   section and time slice**, and click **Explain this spot**.
7. **Close (1 min).** Ask: *Where would you drill next, and what could prove you wrong?*

If the shared model is slow on the day, steps 2, 4 and the tables in **How it works** need no model at all.

---

## If something goes wrong

Type this in the terminal, inside the `exploration-copilot` folder, and send me everything it prints:

```
.venv/bin/python scripts/check.py --data
```

It lists what is installed and how the data files are laid out (file names and column names only, no data
values). If a run stopped, also send the last lines of the log:

```
tail -n 40 workspace/logs/pipeline.log
```

To start the interpretation again from scratch, click **Re-run interpretation** in the app.
