"""Process-local numerical thread limits; never change host configuration."""
import os


def cpu_threads(environ=None):
    environ = os.environ if environ is None else environ
    try:
        requested = max(1, min(24, int(environ.get("COPILOT_CPU_THREADS", "4"))))
    except ValueError:
        requested = 4
    if environ.get("SLURM_JOB_ID"):
        try:
            allocated = max(1, int(environ.get("SLURM_CPUS_PER_TASK", "1")))
        except ValueError:
            allocated = 1
        requested = min(requested, allocated)
    return requested


def configure_threads():
    # One BLAS thread per calling thread prevents nested BLAS/OpenMP pool growth.
    # Override inherited machine-wide settings only inside this app process.
    for variable in ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "MKL_NUM_THREADS",
                     "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = "1"
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads())
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads())
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
