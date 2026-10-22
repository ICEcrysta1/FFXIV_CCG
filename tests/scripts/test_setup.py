"""隔离外部命令，验证安装入口的虚拟环境与 Node 运行时恢复。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

# 在安装依赖时停止或使用替身，测试不执行联网、系统安装或 GPU 依赖下载。
DEPENDENCY_INSTALL_REACHED = 73
NODE_DEPENDENCY_INSTALL_REACHED = 74
FAKE_COMMAND = r'''
import json
import shutil
import sys
from pathlib import Path

root = Path.cwd()
command = Path(sys.argv[0])
arguments = sys.argv[1:]
with (root / "calls.jsonl").open("a", encoding="utf-8") as stream:
    stream.write(json.dumps([str(command), arguments]) + "\n")

if command.name == "uname":
    print("Linux" if arguments == ["-s"] else (root / "host_arch").read_text())
elif command.name == "curl":
    if (root / "download_failed").exists():
        sys.exit(42)
    shutil.copyfile(root / "node-download" / arguments[-3].rsplit("/", 1)[-1], arguments[-1])
elif command.name == "node":
    metadata = json.loads((command.parent.parent / "runtime.json").read_text())
    if metadata.get("exit_code"):
        sys.exit(metadata["exit_code"])
    if arguments[0] == "-p":
        print("/".join([metadata["platform"], metadata["arch"], metadata["version"].split(".")[0]]))
    elif arguments == ["--version"]:
        print("v" + metadata["version"])
    else:
        sys.exit(74)
elif command.name == "apt-get":
    if (root / "apt_failed").exists():
        sys.exit(42)
    if arguments[0] == "install":
        (root / "ensurepip_ready").touch()
elif command.name.startswith("python3"):
    environment = root / ".venv"
    in_environment = command.parent == environment / "bin"
    version = "3.12" if command.name == "python3.12" else "3.14"
    if arguments[0] == "-c":
        code = arguments[1]
        if "sysconfig.get_path" in code and (root / "python_headers_missing").exists():
            print(f"缺少 Python 开发头文件：/usr/include/python{version}/Python.h，请先安装 python{version}-dev 后重新运行。", file=sys.stderr)
            sys.exit(1)
        if "import venv, ensurepip" in code:
            sys.exit(0 if (root / "ensurepip_ready").exists() else 1)
        if "sys._base_executable" in code:
            version_file = environment / "base_python"
            print(root / "commands" / (version_file.read_text() if version_file.exists() else "python3"))
        elif "print(f" in code:
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
        sys.exit(0 if (root / "test_node_setup").exists() else 73)
    elif arguments[:3] == ["-m", "pip", "show"]:
        sys.exit(1)
    else:
        sys.exit(99)
'''


class SetupTests(unittest.TestCase):
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
        for name in ("dirname", "awk"):
            executable = shutil.which(name)
            self.assertIsNotNone(executable)
            (self.commands / name).symlink_to(executable)
        for name in ("python3", "python3.12", "rustc", "cargo", "cc", "apt-get", "git", "curl", "tar", "xz", "sha256sum", "uname", "node"):
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

    def prepare_node_runtime(self, runtime, *, platform="linux", arch="x64", version="22.0.0", exit_code=0):
        (runtime / "bin").mkdir(parents=True, exist_ok=True)
        (runtime / "runtime.json").write_text(json.dumps({
            "platform": platform, "arch": arch, "version": version, "exit_code": exit_code,
        }), encoding="utf-8")
        shutil.copyfile(self.commands / "node", runtime / "bin/node")
        (runtime / "bin/node").chmod(0o755)
        return runtime

    def prepare_node_setup(self, *, host_arch="x86_64"):
        self.prepare_environment(pip_ready=True)
        (self.root / "test_node_setup").touch()
        (self.root / "host_arch").write_text(host_arch, encoding="utf-8")
        (self.root / ".node-version").write_text("22\n", encoding="utf-8")
        # 下载使用本地替身，校验、解压及替换仍走真实工具。
        for name in ("tr", "mkdir", "mktemp", "mv", "rm", "tar", "xz", "sha256sum"):
            executable = shutil.which(name)
            self.assertIsNotNone(executable)
            (self.commands / name).unlink(missing_ok=True)
            (self.commands / name).symlink_to(executable)
        arch = "x64" if host_arch == "x86_64" else "arm64"
        archive_root = f"node-v22.0.0-linux-{arch}"
        download = self.root / "node-download"
        download.mkdir()
        source = self.prepare_node_runtime(download / archive_root, arch=arch)
        archive = download / f"{archive_root}.tar.xz"
        with tarfile.open(archive, "w:xz") as stream:
            stream.add(source, arcname=archive_root)
        checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
        (download / "SHASUMS256.txt").write_text(f"{checksum}  {archive.name}\n", encoding="utf-8")
        cache = self.root / ".node/npm-cache/keep"
        cache.parent.mkdir(parents=True)
        cache.touch()
        return self.root / ".node/runtime"

    def assert_node_ready(self, result, calls, *, downloaded):
        self.assertEqual(result.returncode, NODE_DEPENDENCY_INSTALL_REACHED, result.stdout + result.stderr)
        self.assertEqual(calls[-1][0], str(self.root / ".node/runtime/bin/node"))
        self.assertIn("install", calls[-1][1])
        downloads = [arguments for command, arguments in calls if Path(command).name == "curl"]
        self.assertEqual(len(downloads), 2 if downloaded else 0)
        self.assertTrue((self.root / ".node/npm-cache/keep").exists())
        self.assertEqual(list((self.root / ".node").glob("stage.*")), [])

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

    def test_missing_python_headers_report_selected_interpreter_before_installation(self):
        (self.root / "python_headers_missing").touch()
        for interpreter, version in (("python3", "3.14"), ("python3.12", "3.12")):
            with self.subTest(interpreter=interpreter):
                if interpreter == "python3.12":
                    environment = self.prepare_environment(pip_ready=True)
                    (environment / "base_python").write_text(interpreter)
                (self.root / "calls.jsonl").unlink(missing_ok=True)
                result, calls = self.run_setup()
                self.assertEqual(result.returncode, 1)
                self.assertIn("Python.h", result.stderr)
                self.assertIn(f"python{version}-dev", result.stderr)
                self.assertEqual(Path(calls[-1][0]).name, interpreter)
                self.assertFalse(any(Path(command).name == "apt-get" or arguments[:1] == ["-m"]
                                     for command, arguments in calls))

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

    def test_reuses_compatible_node_runtime(self):
        runtime = self.prepare_node_setup()
        self.prepare_node_runtime(runtime, version="22.3.1")
        marker = runtime / "keep"
        marker.touch()
        result, calls = self.run_setup()
        self.assert_node_ready(result, calls, downloaded=False)
        self.assertTrue(marker.exists())

    def test_reinstalls_windows_node_runtime(self):
        runtime = self.prepare_node_setup()
        runtime.mkdir()
        (runtime / "node.exe").write_bytes(b"Windows Node")
        result, calls = self.run_setup()
        self.assert_node_ready(result, calls, downloaded=True)
        self.assertFalse((runtime / "node.exe").exists())
        self.assertTrue((runtime / "bin/node").is_file())

    def test_reinstalls_incompatible_node_runtime(self):
        runtime = self.prepare_node_setup()
        for metadata in (
            {"platform": "darwin"},
            {"arch": "arm64"},
            {"version": "20.0.0"},
            {"exit_code": 126},
        ):
            with self.subTest(metadata=metadata):
                shutil.rmtree(runtime, ignore_errors=True)
                (self.root / "calls.jsonl").unlink(missing_ok=True)
                self.prepare_node_runtime(runtime, **metadata)
                (runtime / "stale").touch()
                result, calls = self.run_setup()
                self.assert_node_ready(result, calls, downloaded=True)
                self.assertFalse((runtime / "stale").exists())

    def test_installs_missing_or_incomplete_node_runtime(self):
        runtime = self.prepare_node_setup()
        for state in ("missing", "empty", "not_executable"):
            with self.subTest(state=state):
                shutil.rmtree(runtime, ignore_errors=True)
                (self.root / "calls.jsonl").unlink(missing_ok=True)
                if state == "empty":
                    runtime.mkdir()
                elif state == "not_executable":
                    self.prepare_node_runtime(runtime)
                    (runtime / "bin/node").chmod(0o644)
                result, calls = self.run_setup()
                self.assert_node_ready(result, calls, downloaded=True)

    def test_replaces_x64_runtime_on_arm64_host_and_reuses_on_retry(self):
        runtime = self.prepare_node_setup(host_arch="aarch64")
        self.prepare_node_runtime(runtime, arch="x64")
        result, calls = self.run_setup()
        self.assert_node_ready(result, calls, downloaded=True)
        self.assertEqual(json.loads((runtime / "runtime.json").read_text())["arch"], "arm64")
        (self.root / "calls.jsonl").unlink()
        (self.root / "host_arch").write_text("arm64", encoding="utf-8")
        result, calls = self.run_setup()
        self.assert_node_ready(result, calls, downloaded=False)

    def test_node_download_failure_keeps_existing_runtime(self):
        runtime = self.prepare_node_setup()
        runtime.mkdir()
        previous = runtime / "node.exe"
        previous.write_bytes(b"Windows Node")
        (self.root / "download_failed").touch()
        result, calls = self.run_setup()
        self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
        self.assertEqual(previous.read_bytes(), b"Windows Node")
        self.assertEqual(sum(Path(command).name == "curl" for command, _ in calls), 1)
        self.assertFalse(any(Path(command).name == "node" for command, _ in calls))
        self.assertEqual(list((self.root / ".node").glob("stage.*")), [])

    def test_node_checksum_failure_keeps_existing_runtime(self):
        runtime = self.prepare_node_setup()
        runtime.mkdir()
        previous = runtime / "node.exe"
        previous.write_bytes(b"Windows Node")
        checksums = self.root / "node-download/SHASUMS256.txt"
        archive_name = checksums.read_text().split()[1]
        checksums.write_text(f"{'0' * 64}  {archive_name}\n", encoding="utf-8")
        result, calls = self.run_setup()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(f"{archive_name}: FAILED", result.stdout)
        self.assertEqual(previous.read_bytes(), b"Windows Node")
        self.assertEqual(sum(Path(command).name == "curl" for command, _ in calls), 2)
        self.assertFalse(any(Path(command).name == "node" for command, _ in calls))
        self.assertEqual(list((self.root / ".node").glob("stage.*")), [])


if __name__ == "__main__":
    unittest.main()
