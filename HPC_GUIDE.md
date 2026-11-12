# Getting Exploration Copilot running on the HPC

Follow the parts in order. Type each command exactly, one line at a time, and press Enter after each.
Tip: you can copy a line here and paste it into the HPC terminal with **Ctrl + Shift + V**.

---

## Part A — Put the app on GitHub (on your laptop, about 5 minutes)

You need this anyway: the hackathon asks for a public GitHub repository.

1. Unzip `exploration-copilot.zip` (right-click it → **Extract All**). You get a folder called `exploration-copilot`.
2. Go to **github.com** and sign in (create a free account if you have none).
3. Click the **+** at the top right → **New repository**.
4. Repository name: `exploration-copilot`. Choose **Public**. Click **Create repository**.
5. On the next page click the link **uploading an existing file**.
6. Open the `exploration-copilot` folder on your laptop. Press **Ctrl + A** to select everything inside it,
   and drag it all onto the GitHub page.
7. Wait until the files are listed, then click the green **Commit changes** button.

Write down your GitHub user name. Below it is shown as `YOURNAME`.

---

## Part B — Install on the HPC (once, about 20 minutes, mostly waiting)

1. Connect with RustDesk and log in, as in the organisers' manual.
2. Open a terminal: top-left menu **Applications → System Tools → Terminal**.
3. Type these four lines (replace `YOURNAME`):

```
cd /data/epic20/APGCE/$USER
git clone https://github.com/YOURNAME/exploration-copilot.git
cd exploration-copilot
bash setup.sh
```

4. Wait. Text scrolls by while it installs and downloads the models. It ends with a list called
   **Self-check**. `[OK]` means ready. `[--]` means that piece is missing; the app still works with a
   simpler method for it, but send me the list so I can help.

If a line fails:

| What you see | What to type instead |
|---|---|
| `cd: /data/epic20/APGCE/...: No such file or directory` | `ls /data/epic20/APGCE/` to see your team's folder name, then `cd /data/epic20/APGCE/that_name` |
| `git: command not found` | `curl -L -o ec.zip https://github.com/YOURNAME/exploration-copilot/archive/refs/heads/main.zip` then `unzip ec.zip` then `cd exploration-copilot-main` then `bash setup.sh` |
| GitHub cannot be reached at all | In RustDesk use the **file transfer** button to copy `exploration-copilot.zip` to the HPC, then `unzip exploration-copilot.zip`, `cd exploration-copilot`, `bash setup.sh` |

---

## Part C — Start the app (every time)

In the terminal, inside the `exploration-copilot` folder:

```
bash start.sh
```

Firefox opens by itself. If it does not, open Firefox and type `http://127.0.0.1:8050` in the address bar.
Keep the terminal window open while you use the app.

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
it does not start/cancel jobs, query system resources, install anything, or modify data. It starts
following logs from their current ends. If `COPILOT_WORKSPACE` was set for your app, use the same
setting here (or pass `--workspace /your/app/workspace`).

Logs remain inside `workspace/logs/` (or your configured workspace):

* `setup.log`: append-only setup output, including failures.
* `server.log`: append-only server output and terminal monitoring from `bash start.sh`.
* `pipeline.log`: current interpretation output.
* `sdip_*.log`: live converter output, available before a conversion finishes.
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

---

## Part D — First run (once — do this well before the judging)

1. Click **Import data** (top right). The challenge folder is already filled in. Click **Use this folder**.
2. Click **Run interpretation**. A progress bar shows the steps. The first step (converting the nine
   volumes) is the slowest. You can browse the seismic as soon as the first volume is ready.
3. Before you click Run, close any Jupyter notebook that uses the **GPU via SLURM** kernel. Each team gets
   one GPU job at a time, and the app needs it.
4. When the bar says **Finished**, everything is saved. From now on the app opens ready to use.

---

## Part E — Using it

**Left side — the viewer**

* **Inline / Crossline** tabs switch the section. **Time slice / Horizon map** tabs switch the map.
* Click anywhere to move the crosshair; both pictures follow.
* Scroll the mouse wheel to step through lines (or through time on the map). Arrow keys work too.
* **Ctrl + scroll** zooms, drag moves, double-click resets.
* **Volume** picks which volume is shown (full, near, far, impedance, sand probability…).
* The round buttons switch layers on and off: Horizons, Faults, Facies, Wells, Leads.
* **Find look-alikes** highlights places whose seismic resembles the crosshair spot.
* **Explain this spot** sends the assistant a picture of what you are looking at.

**Right side — the assistant**

* **Assistant** tab: ask in plain words. Blue tags like `E7` are facts — hover to read, click to jump there.
  Yellow tags jump to a location. Under each answer: **Show the facts used** and **Challenge this**.
* **Leads** tab: the ranked traps with their scores. **Show on map**, **Explain**, **Argue both sides**.
* **How it works** tab: which methods ran, the well checks, and the limits of the result.

---

## Part F — A 5-minute demo for the judges

1. Scroll through a few inlines, switch to crossline, show the crosshair and a time slice. Switch the volume to
   sand probability. Turn Faults on.
2. Open **Leads**. Click **Show on map** on the first lead.
3. Click **Explain**. While it answers, point out the grey lines showing what it looked at.
4. Click a blue fact tag: the viewer jumps to the evidence.
5. Click **Challenge this**: the assistant argues against its own answer and gives a verdict.
6. Put the crosshair somewhere interesting and click **Explain this spot**, then **Find look-alikes**.
7. Ask: *Where would you drill next, and what could prove you wrong?*

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
