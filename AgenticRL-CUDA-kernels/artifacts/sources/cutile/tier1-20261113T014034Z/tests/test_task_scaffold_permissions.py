import hashlib
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import utils.task_suite as task_suite


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ScaffoldPermissionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="task-scaffold-readonly-")
        self.root = Path(self._tmp.name) / "case"
        self.task = self.root / "tasks" / "row_softmax"
        self.task.mkdir(parents=True)
        files = {
            "binding.cpp": "// synthetic binding\n",
            "binding_registry.h": "// synthetic registry\n",
        }
        for name, text in files.items():
            (self.root / name).write_text(text)
        self.task_files = {
            "model.py": "# synthetic protected reference\n",
            "model_new.py": "import model_new_triton\nfrom model_new_triton import ModelNew\n\nBACKEND = \"triton\"\n",
            "model_new_triton.py": "# synthetic honest fixture\nBACKEND = \"triton\"\n",
            "model_new_hack.py": "# synthetic hack fixture\nBACKEND = \"triton\"\n",
        }
        for name, text in self.task_files.items():
            path = self.task / name
            path.write_text(text)
            path.chmod(0o444)
        for name in files:
            (self.root / name).chmod(0o444)
        self._patch = patch.object(task_suite, "ROOT", self.root)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.addCleanup(self._cleanup_readonly)

    def _cleanup_readonly(self):
        for path in self.root.rglob("*"):
            if path.is_file():
                os.chmod(path, 0o644)
        self._tmp.cleanup()

    def test_scaffold_overwrites_readonly_alias(self):
        for backend, variant in (("triton", "honest"), ("cuda", "hack")):
            with self.subTest(backend=backend, variant=variant):
                dest = Path(self._tmp.name) / f"dest-{backend}-{variant}"
                task_suite.scaffold("row_softmax", backend, variant, dest)
                alias = dest / "model_new.py"
                self.assertTrue(alias.is_file())
                self.assertTrue(os.stat(alias).st_mode & stat.S_IWUSR)
                if variant == "honest":
                    self.assertEqual(alias.read_text(), self.task_files["model_new_triton.py"])
                else:
                    self.assertIn("BACKEND = 'cuda'", alias.read_text())
                    self.assertNotIn('BACKEND = "triton"', alias.read_text())
                for name in ("model.py", "model_new_triton.py", "model_new_hack.py"):
                    self.assertEqual(_sha256(dest / name), _sha256(self.task / name))
                    self.assertEqual(stat.S_IMODE(os.stat(self.task / name).st_mode), 0o444)
                for name in ("binding.cpp", "binding_registry.h"):
                    self.assertEqual(_sha256(dest / name), _sha256(self.root / name))
        for name in self.task_files:
            self.assertEqual(stat.S_IMODE(os.stat(self.task / name).st_mode), 0o444)


if __name__ == "__main__":
    unittest.main()
