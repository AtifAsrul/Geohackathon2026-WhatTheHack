"""Download the official Precision First TorchScript model; run on the HPC once."""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from copilot import config

# Linked as Precision First in https://github.com/douyimin/FaultSSL
PRECISION_FIRST_ID = "1_LwU0TOhssAqYO3Kp5wLxtPFuE3lv7f9"


def valid_checkpoint(path):
    try:
        with zipfile.ZipFile(path) as archive:
            return any("/code/" in name for name in archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False


def main():
    out = config.FAULTSSL_WEIGHTS
    if valid_checkpoint(out):
        print(f"FaultSSL Precision First already present: {out}")
        return 0
    try:
        import gdown
    except ImportError:
        print("Install the download helper first: .venv/bin/python -m pip install gdown")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".download")
    try:
        gdown.download(id=PRECISION_FIRST_ID, output=str(tmp), quiet=False)
        if not valid_checkpoint(tmp):
            raise ValueError("The download is not a FaultSSL TorchScript archive (it may be a Drive error page).")
        tmp.replace(out)
    except Exception as e:
        print(f"FaultSSL download failed: {e}")
        return 1
    print(f"FaultSSL Precision First ready: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
