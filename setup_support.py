"""Low-memory, resumable Windows setup helpers. Uses only the standard library."""
import argparse
import ctypes
import hashlib
from html.parser import HTMLParser
import http.client
import importlib.metadata
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
GIB = 1024 ** 3
TORCH_VERSION = "2.11.0"


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong) for name in (
            "total_physical", "available_physical", "total_commit", "available_commit",
            "total_virtual", "available_virtual", "extended")]


def resource_failures(free_ram, free_commit, free_disk, stage):
    failures = []
    if free_ram < 3 * GIB:
        failures.append("at least 3 GiB of available RAM is required")
    if free_commit < 4 * GIB:
        failures.append("at least 4 GiB of available Windows commit capacity is required")
    if free_disk < (12 if stage == "setup" else 5) * GIB:
        failures.append(f"at least {12 if stage == 'setup' else 5} GiB of free disk space is required")
    return failures


def preflight(stage):
    if sys.platform != "win32":
        raise RuntimeError("This setup helper is for Windows; use the README's manual setup on other systems.")
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError()
    disk = shutil.disk_usage(ROOT).free
    print(f"Available: RAM {memory.available_physical / GIB:.2f} GiB; "
          f"Windows commit capacity {memory.available_commit / GIB:.2f} GiB; "
          f"disk {disk / GIB:.2f} GiB.", flush=True)
    failures = resource_failures(memory.available_physical, memory.available_commit, disk, stage)
    if failures:
        raise RuntimeError("Preflight stopped: " + "; ".join(failures) +
                           ". Close unused apps/tabs or restart Windows, then rerun the same command. "
                           "Your existing environment and downloads are preserved.")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def resolve_wheel(flavor):
    index = f"https://download.pytorch.org/whl/{flavor}/torch/"
    filename = f"torch-{TORCH_VERSION}+{flavor}-cp312-cp312-win_amd64.whl"
    with urllib.request.urlopen(index, timeout=45) as response:
        document = response.read(8 * 1024 * 1024).decode("utf-8")
    return wheel_from_index(index, document, filename)


def wheel_from_index(index, document, filename):
    parser = Links()
    parser.feed(document)
    for href in parser.hrefs:
        url = urllib.parse.urljoin(index, href)
        parsed = urllib.parse.urlsplit(url)
        if urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1]) != filename:
            continue
        if parsed.scheme != "https" or parsed.netloc not in ("download.pytorch.org", "download-r2.pytorch.org"):
            raise ValueError("Unexpected host for the PyTorch wheel.")
        digest = urllib.parse.parse_qs(parsed.fragment).get("sha256", [""])[0]
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("The PyTorch index did not supply a SHA-256 hash.")
        # The index also lists the r2 mirror, which can return 403 for public
        # wheels. The primary host serves the same path and published digest.
        return urllib.parse.urlunsplit(parsed._replace(netloc="download.pytorch.org", fragment="")), filename, digest
    raise ValueError(f"Cannot find {filename} on the official PyTorch index.")


def download(url, target, digest, attempts=4, opener=urllib.request.urlopen):
    """Stream to disk and retain .part on interruption; verify before promotion."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if target.exists():
        if sha256_file(target) == digest:
            print("Using verified cached wheel.", flush=True)
            return
        raise ValueError(f"Cached wheel failed its hash check: {target}. Rename it before retrying.")
    # The process may have stopped after receiving the last byte but before rename.
    if partial.exists() and sha256_file(partial) == digest:
        partial.replace(target)
        return
    for attempt in range(attempts):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        print(f"Downloading wheel (attempt {attempt + 1}/{attempts}, resuming at {offset / 1e6:.1f} MB).", flush=True)
        try:
            request = urllib.request.Request(url, headers=headers)
            with opener(request, timeout=45) as response:
                status = response.status
                if status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset or not offset <= int(match[2]) < int(match[3]):
                        raise ValueError("Invalid resume response; partial download was preserved.")
                    total = int(match[3])
                    mode = "ab"
                elif status == 200:
                    # A server may ignore Range. Overwrite rather than append a duplicate.
                    if offset:
                        print("Server ignored the resume request; restarting this wheel.", flush=True)
                    offset = 0
                    length = response.headers.get("Content-Length")
                    total = int(length) if length else None
                    mode = "wb"
                else:
                    raise ValueError(f"Unexpected download status {status}.")
                written = offset
                last_report = time.monotonic()
                with partial.open(mode) as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        written += len(chunk)
                        if time.monotonic() - last_report >= 2:
                            print(f"  {written / 1e6:.0f} MB downloaded" +
                                  (f" / {total / 1e6:.0f} MB" if total else ""), flush=True)
                            last_report = time.monotonic()
            if total is not None and written != total:
                raise OSError(f"Connection ended after {written} of {total} bytes.")
            if sha256_file(partial) != digest:
                raise ValueError(f"Downloaded wheel failed its hash check. Rename {partial} before retrying.")
            partial.replace(target)
            print("Download complete; SHA-256 verified.", flush=True)
            return
        except (OSError, urllib.error.URLError, http.client.IncompleteRead) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                # Repeating permission/not-found errors cannot resume a file.
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    raise RuntimeError(
                        f"Download server returned HTTP {exc.code} for {url}. "
                        "This is not a connection interruption; automatic retries stopped. "
                        "Any existing partial file has been preserved.") from exc
            if attempt + 1 == attempts:
                raise RuntimeError(f"Download interrupted: {exc}. Rerun setup to resume {partial}.") from exc
            if isinstance(exc, urllib.error.HTTPError):
                time.sleep(min(2 ** attempt, 8))
            print(f"Connection interrupted: {exc}. Retrying the saved partial file.", flush=True)


def install_torch(flavor):
    if sys.version_info[:2] != (3, 12) or sys.platform != "win32" or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError("Use 64-bit Python 3.12 for this Windows setup.")
    expected = f"{TORCH_VERSION}+{flavor}"
    try:
        if importlib.metadata.version("torch") == expected:
            print(f"PyTorch {expected} is already installed; skipping its download.", flush=True)
            return
    except importlib.metadata.PackageNotFoundError:
        pass
    url, filename, digest = resolve_wheel(flavor)
    target = ROOT / ".cache" / "wheels" / filename
    print(f"Wheel source: {url}", flush=True)
    download(url, target, digest)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", str(target)], check=True)


def check_download(flavor):
    """Verify the real resolver and HTTP Range support using two 64 KiB reads."""
    url, filename, digest = resolve_wheel(flavor)
    print(f"Wheel: {filename}\nSource: {url}\nPublished SHA-256: {digest}", flush=True)
    for start in (0, 65536):
        end = start + 65535
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=45) as response:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if response.status != 206 or not match or int(match[1]) != start or int(match[2]) != end:
                raise RuntimeError("The server did not honor the bounded range request.")
            data = response.read(65536)
            if len(data) != 65536 or (start == 0 and not data.startswith(b"PK")):
                raise RuntimeError("The server did not return the expected wheel bytes.")
            print(f"PASS: downloaded bytes {start}-{end} (HTTP 206).", flush=True)
    print("Download and resume checks passed; no installation was performed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight").add_argument("--stage", choices=("setup", "run"), default="setup")
    sub.add_parser("torch").add_argument("--flavor", choices=("cpu", "cu128"), default="cu128")
    sub.add_parser("check-download").add_argument("--flavor", choices=("cpu", "cu128"), default="cu128")
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            preflight(args.stage)
        elif args.command == "check-download":
            check_download(args.flavor)
        else:
            install_torch(args.flavor)
    except KeyboardInterrupt:
        print("Interrupted. Rerun setup to resume any saved wheel download.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
