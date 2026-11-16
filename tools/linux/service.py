"""Manage only this checkout's Linux web process; reuse it when already running."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "logs/linux-web.json"


def process_stamp(pid):
    try:
        # comm may contain spaces and parentheses; fields after the final ')' start at field 3.
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if os.fsencode(ROOT / "app.py") not in args:
            return None
        return stat[19]  # Linux /proc field 22: process start time.
    except (OSError, IndexError, ValueError):
        return None


def running():
    try:
        state = json.loads(STATE.read_text())
        pid = state["pid"]
        if (type(pid) is int and pid > 1 and isinstance(state.get("stamp"), str)
                and state["stamp"] == process_stamp(pid)):
            return state
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "status"))
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--port", type=int, default=int(os.environ.get("FLOWHMR_PORT", "8080")))
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    STATE.parent.mkdir(exist_ok=True)
    with (STATE.parent / "linux-web.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = running()
        if args.action == "stop":
            if state:
                os.kill(state["pid"], signal.SIGTERM)
                for _ in range(100):
                    if process_stamp(state["pid"]) != state["stamp"]:
                        break
                    time.sleep(0.1)
                else:
                    print("Process is still stopping; retry later.")
                    return 1
            STATE.unlink(missing_ok=True)
            print("FlowHMR stopped.")
            return 0
        if args.action == "status":
            print(f"FlowHMR running: {state['url']} (PID {state['pid']})" if state else "FlowHMR is stopped.")
            return 0 if state else 1
        if not state:
            url = f"http://127.0.0.1:{args.port}"
            command = [sys.executable, str(ROOT / "app.py"), "--host", "127.0.0.1", "--port", str(args.port),
                       "--device", "cuda:0", "--sam-chunk-default", "4"]
            with (ROOT / "logs/linux-web.log").open("ab") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log, start_new_session=True)
            try:
                for _ in range(600):
                    if process.poll() is not None:
                        raise RuntimeError("FlowHMR exited. See logs/linux-web.log.")
                    try:
                        with urlopen(url + "/api/info", timeout=1) as response:
                            ready = response.status == 200 and "device_backend" in json.load(response)
                        if ready:
                            # Ensure this child owns the server after binding, including occupied-port failures.
                            time.sleep(0.3)
                            if process.poll() is None:
                                break
                    except (OSError, ValueError):
                        pass
                    time.sleep(0.1)
                else:
                    raise RuntimeError("Startup timed out. See logs/linux-web.log.")
                stamp = process_stamp(process.pid)
                if stamp is None:
                    raise RuntimeError("FlowHMR exited during startup. See logs/linux-web.log.")
                state = {"pid": process.pid, "stamp": stamp, "url": url}
                temporary = STATE.with_suffix(".tmp")
                temporary.write_text(json.dumps(state))
                temporary.replace(STATE)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
                raise
        print(f"FlowHMR ready: {state['url']}")
        if args.open:
            webbrowser.open(state["url"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
