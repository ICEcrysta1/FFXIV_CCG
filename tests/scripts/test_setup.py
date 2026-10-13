"""隔离外部命令，验证安装入口的虚拟环境创建与失败恢复。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


# 在安装项目依赖前停止，测试不执行联网、系统安装或 GPU 依赖下载。
DEPENDENCY_INSTALL_REACHED = 73
FAKE_COMMAND = r'''
import json
import sys
from pathlib import Path

root = Path.cwd()
command = Path(sys.argv[0])
arguments = sys.argv[1:]
with (root / "calls.jsonl").open("a", encoding="utf-8") as stream:
    stream.write(json.dumps([str(command), arguments]) + "\n")

if command.name == "dotnet":
    print("10.0.100")
elif command.name == "apt-get":
    if (root / "apt_failed").exists():
        sys.exit(42)
    if arguments[0] == "install":
        (root / "ensurepip_ready").touch()
elif command.name.startswith("python3"):
    environment = root / ".venv"
    in_environment = command.parent == environment / "bin"
    if arguments[0] == "-c":
        code = arguments[1]
        if "import venv, ensurepip" in code:
            sys.exit(0 if (root / "ensurepip_ready").exists() else 1)
        if "sys._base_executable" in code:
            version_file = environment / "base_python"
            print(root / "commands" / (version_file.read_text() if version_file.exists() else "python3"))
        elif "print(f" in code:
            version = "3.12" if command.name == "python3.12" else "3.14"
            print(f"python{version}-venv")
        elif "sys.prefix" in code:
            if not in_environment or not (environment / "pyvenv.cfg").exists():
                print(".venv 需要有效的 Python 3.12+ 虚拟环境", file=sys.stderr)
                sys.exit(1)
    elif arguments[:2] == ["-m", "venv"]:
        if not (root / "ensurepip_ready").exists():
            sys.exit(1)
        (environment / "bin").mkdir(parents=True, exist_ok=True)
        interpreter = environment / "bin/python3"
        if not interpreter.exists():
            interpreter.symlink_to(command)
        (environment / "pyvenv.cfg").touch()
        (environment / "pip_ready").touch()
    elif arguments[:3] == ["-m", "pip", "--version"]:
        sys.exit(0 if (environment / "pip_ready").exists() else 1)
    elif arguments[:3] == ["-m", "pip", "install"]:
        sys.exit(73)
    else:
        sys.exit(99)
'''


class SetupVirtualEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.commands = self.root / "commands"
        self.commands.mkdir()
        project = Path(__file__).resolve().parents[2]
        self.setup = self.root / "setup.sh"
        self.setup.write_bytes((project / "setup.sh").read_bytes())
        self.bash = shutil.which("bash")
        self.assertIsNotNone(self.bash)
        for name in ("dirname", "uname", "awk"):
            executable = shutil.which(name)
            self.assertIsNotNone(executable)
            (self.commands / name).symlink_to(executable)
        for name in ("python3", "python3.12", "dotnet", "apt-get", "git", "curl", "tar", "xz", "sha256sum"):
            path = self.commands / name
            path.write_text(f"#!{sys.executable}\n" + FAKE_COMMAND, encoding="utf-8")
            path.chmod(0o755)

    def prepare_environment(self, *, pip_ready=False):
        environment = self.root / ".venv"
        (environment / "bin").mkdir(parents=True)
        (environment / "bin/python3").symlink_to(self.commands / "python3")
        (environment / "pyvenv.cfg").touch()
        if pip_ready:
            (environment / "pip_ready").touch()
        return environment

    def run_setup(self):
        # PATH 只含替身，避免测试意外调用真实 apt-get 或下载工具。
        result = subprocess.run(
            [self.bash, str(self.setup)], cwd=self.root,
            env={**os.environ, "PATH": str(self.commands)},
            capture_output=True, text=True, check=False,
        )
        calls = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
        return result, calls

    def assert_ready(self, result, calls):
        self.assertEqual(result.returncode, DEPENDENCY_INSTALL_REACHED, result.stdout + result.stderr)
        self.assertEqual(calls[-1], [str(self.root / ".venv/bin/python3"), ["-m", "pip", "install", "--upgrade", "pip"]])

    def test_creates_missing_environment(self):
        (self.root / "ensurepip_ready").touch()
        result, calls = self.run_setup()
        self.assert_ready(result, calls)
        self.assertTrue((self.root / ".venv/pip_ready").exists())
        self.assertFalse(any(Path(command).name == "apt-get" for command, _ in calls))

    def test_repairs_missing_pip_without_removing_existing_packages(self):
        environment = self.prepare_environment()
        existing_package = environment / "existing_package"
        existing_package.write_text("保留已有依赖", encoding="utf-8")
        (self.root / "ensurepip_ready").touch()
        result, calls = self.run_setup()
        self.assert_ready(result, calls)
        self.assertEqual(existing_package.read_text(encoding="utf-8"), "保留已有依赖")
        self.assertTrue((environment / "pip_ready").exists())

    def test_repairs_directory_without_interpreter(self):
        (self.root / ".venv").mkdir()
        (self.root / "ensurepip_ready").touch()
        result, calls = self.run_setup()
        self.assert_ready(result, calls)

    @unittest.skipUnless(os.geteuid() == 0, "此用例覆盖 root 自动安装分支")
    def test_installs_matching_venv_package_and_recovers_on_retry(self):
        self.prepare_environment()
        result, calls = self.run_setup()
        self.assert_ready(result, calls)
        apt_calls = [arguments for command, arguments in calls if Path(command).name == "apt-get"]
        self.assertEqual(apt_calls, [["update"], ["install", "-y", "python3.14-venv"]])
        (self.root / "calls.jsonl").unlink()
        result, calls = self.run_setup()
        self.assert_ready(result, calls)
        self.assertFalse(any(Path(command).name == "apt-get" or arguments[:2] == ["-m", "venv"] for command, arguments in calls))

    @unittest.skipUnless(os.geteuid() == 0, "此用例覆盖 root 自动安装分支")
    def test_repair_uses_existing_environment_python_version(self):
        environment = self.prepare_environment()
        (environment / "base_python").write_text("python3.12")
        result, calls = self.run_setup()
        self.assert_ready(result, calls)
        self.assertIn([str(self.commands / "apt-get"), ["install", "-y", "python3.12-venv"]], calls)
        self.assertIn([str(self.commands / "python3.12"), ["-m", "venv", ".venv"]], calls)

    def test_valid_environment_is_reused_without_venv_support(self):
        self.prepare_environment(pip_ready=True)
        result, calls = self.run_setup()
        self.assert_ready(result, calls)
        self.assertFalse(any(Path(command).name == "apt-get" or arguments[:2] == ["-m", "venv"] for command, arguments in calls))

    def test_missing_package_manager_reports_manual_install(self):
        (self.commands / "apt-get").unlink()
        result, calls = self.run_setup()
        self.assertEqual(result.returncode, 1)
        self.assertIn("python3.14-venv", result.stderr)
        self.assertFalse(any(arguments[:2] == ["-m", "venv"] for _, arguments in calls))

    @unittest.skipUnless(os.geteuid() == 0, "此用例覆盖 root 自动安装分支")
    def test_package_install_failure_stops_before_dependencies(self):
        (self.root / "apt_failed").touch()
        result, calls = self.run_setup()
        self.assertEqual(result.returncode, 42)
        self.assertFalse(any(arguments[:2] == ["-m", "venv"] for _, arguments in calls))

    def test_rejects_windows_environment(self):
        (self.root / ".venv/Scripts").mkdir(parents=True)
        result, calls = self.run_setup()
        self.assertEqual(result.returncode, 1)
        self.assertIn("不是有效的 Linux 虚拟环境", result.stderr)
        self.assertFalse(any(Path(command).name == "apt-get" for command, _ in calls))

    def test_rejects_interpreter_outside_virtual_environment(self):
        environment = self.prepare_environment()
        (environment / "pyvenv.cfg").unlink()
        result, calls = self.run_setup()
        self.assertEqual(result.returncode, 1)
        self.assertIn("有效的 Python", result.stderr)
        self.assertFalse(any(Path(command).name == "apt-get" for command, _ in calls))


if __name__ == "__main__":
    unittest.main()
