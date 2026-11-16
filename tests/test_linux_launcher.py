"""Linux launcher environment isolation and process ownership checks."""
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux launcher")
class LinuxLauncherTest(unittest.TestCase):
    def test_common_isolates_imports_and_caches_in_unicode_checkout(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "软件盘/FlowHMR Linux"
            common = root / "tools/linux/common.sh"
            common.parent.mkdir(parents=True)
            shutil.copyfile(ROOT / "tools/linux/common.sh", common)
            env = {**os.environ, "PYTHONPATH": "/unrelated/ros", "PYTHONHOME": "/unrelated/python"}
            env.pop("FLOWHMR_ATTENTION_BACKEND", None)
            env.pop("FLOWHMR_ATTENTION_QUERY_CHUNK_SIZE", None)
            code = ('import os,json,sys; print(json.dumps(dict(root=os.getcwd(),'
                    'path=os.environ.get("PYTHONPATH"),home=os.environ.get("PYTHONHOME"),'
                    'utf8=sys.flags.utf8_mode,cache=os.environ["HF_HOME"],'
                    'attention=os.environ["FLOWHMR_ATTENTION_BACKEND"])))')
            child = subprocess.run(["bash", "-c", 'source "$1"; "$2" -c "$3"',
                                    "launcher-test", str(common), sys.executable, code],
                                   env=env, check=True, capture_output=True, text=True, encoding="utf-8")
            result = json.loads(child.stdout)
            self.assertEqual(result["root"], str(root))
            self.assertIsNone(result["path"])
            self.assertIsNone(result["home"])
            self.assertEqual(result["utf8"], 1)
            self.assertEqual(result["cache"], str(root / ".cache/huggingface"))
            self.assertEqual(result["attention"], "query_batched")

    def test_service_rejects_reused_pid_and_unrelated_process(self):
        spec = spec_from_file_location("linux_service_test", ROOT / "tools/linux/service.py")
        service = module_from_spec(spec)
        spec.loader.exec_module(service)
        with TemporaryDirectory() as temporary:
            state = Path(temporary) / "state.json"
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)",
                                      str(ROOT / "app.py")])
            try:
                stamp = service.process_stamp(child.pid)
                self.assertIsNotNone(stamp)
                self.assertIsNone(service.process_stamp(os.getpid()))
                with patch.object(service, "STATE", state):
                    state.write_text(json.dumps({"pid": child.pid, "stamp": stamp, "url": "local"}))
                    self.assertEqual(service.running()["pid"], child.pid)
                    state.write_text(json.dumps({"pid": child.pid, "stamp": "reused-pid", "url": "local"}))
                    self.assertIsNone(service.running())
            finally:
                child.terminate()
                child.wait(timeout=5)
