import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from setup_support import GIB, download, resource_failures, wheel_from_index


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(data))}


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.directory.cleanup)
        self.target = Path(self.directory.name) / "test.whl"
        self.partial = self.target.with_suffix(".whl.part")
        self.data = b"verified wheel contents"
        self.digest = hashlib.sha256(self.data).hexdigest()

    def test_interruption_preserves_progress_and_next_run_resumes(self):
        class Interrupted(Response):
            def read(self, count):
                if self.tell():
                    raise KeyboardInterrupt()
                return super().read(5)
        with self.assertRaises(KeyboardInterrupt):
            download("https://example.test/file", self.target, self.digest,
                     opener=lambda *a, **k: Interrupted(self.data))
        self.assertEqual(self.partial.read_bytes(), self.data[:5])
        self.assertFalse(self.target.exists())

        def resume(request, timeout):
            self.assertEqual(request.get_header("Range"), "bytes=5-")
            return Response(self.data[5:], 206, {"Content-Range": f"bytes 5-{len(self.data)-1}/{len(self.data)}"})
        download("https://example.test/file", self.target, self.digest, opener=resume)
        self.assertEqual(self.target.read_bytes(), self.data)
        self.assertFalse(self.partial.exists())

    def test_server_ignoring_range_restarts_without_duplicating_bytes(self):
        self.partial.write_bytes(self.data[:5])
        download("https://example.test/file", self.target, self.digest,
                 opener=lambda *a, **k: Response(self.data))
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_incorrect_range_is_rejected_without_changing_partial(self):
        self.partial.write_bytes(self.data[:5])
        with self.assertRaisesRegex(ValueError, "Invalid resume"):
            download("https://example.test/file", self.target, self.digest,
                     opener=lambda *a, **k: Response(self.data[4:], 206, {"Content-Range": f"bytes 4-{len(self.data)-1}/{len(self.data)}"}))
        self.assertEqual(self.partial.read_bytes(), self.data[:5])

    def test_corrupt_download_never_becomes_installable_wheel(self):
        with self.assertRaisesRegex(ValueError, "hash check"):
            download("https://example.test/file", self.target, self.digest,
                     opener=lambda *a, **k: Response(b"wrong contents"))
        self.assertFalse(self.target.exists())
        self.assertTrue(self.partial.exists())

    def test_short_read_retries_from_saved_offset(self):
        requests = []
        def opener(request, timeout):
            requests.append(request.get_header("Range"))
            if len(requests) == 1:
                return Response(self.data[:5], headers={"Content-Length": str(len(self.data))})
            return Response(self.data[5:], 206, {"Content-Range": f"bytes 5-{len(self.data)-1}/{len(self.data)}"})
        download("https://example.test/file", self.target, self.digest, opener=opener)
        self.assertEqual(requests, [None, "bytes=5-"])
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_completed_partial_promoted_without_network(self):
        self.partial.write_bytes(self.data)
        def no_network(*a, **k):
            self.fail("A complete verified partial should not be downloaded again")
        download("https://example.test/file", self.target, self.digest, opener=no_network)
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_verified_cache_reused_without_network(self):
        self.target.write_bytes(self.data)
        def no_network(*a, **k):
            self.fail("Verified cached wheel should not be downloaded again")
        download("https://example.test/file", self.target, self.digest, opener=no_network)

    def test_low_memory_blocks_setup_even_with_enough_disk(self):
        failures = resource_failures(0.7 * GIB, 1.3 * GIB, 24 * GIB, "setup")
        self.assertEqual(len(failures), 2)
        self.assertEqual(resource_failures(4 * GIB, 5 * GIB, 24 * GIB, "setup"), [])

    def test_forbidden_stops_after_one_request_and_preserves_partial(self):
        self.partial.write_bytes(self.data[:5])
        calls = []
        def forbidden(request, timeout):
            calls.append(request.full_url)
            raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)
        with self.assertRaisesRegex(RuntimeError, "HTTP 403"):
            download("https://example.test/file", self.target, self.digest, opener=forbidden)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.partial.read_bytes(), self.data[:5])

    def test_transient_server_error_is_retried(self):
        calls = []
        def transient(request, timeout):
            calls.append(request.full_url)
            if len(calls) == 1:
                raise urllib.error.HTTPError(request.full_url, 503, "Unavailable", {}, None)
            return Response(self.data)
        with patch("setup_support.time.sleep"):
            download("https://example.test/file", self.target, self.digest, opener=transient)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_r2_index_link_uses_primary_host_preserving_path_and_digest(self):
        index = "https://download.pytorch.org/whl/cu128/torch/"
        filename = "torch-2.11.0+cu128-cp312-cp312-win_amd64.whl"
        path = "/whl/cu128/" + filename.replace("+", "%2B")
        html = f'<a href="https://download-r2.pytorch.org{path}#sha256={self.digest}">wheel</a>'
        url, actual_name, digest = wheel_from_index(index, html, filename)
        self.assertEqual(url, "https://download.pytorch.org" + path)
        self.assertEqual((actual_name, digest), (filename, self.digest))

    def test_index_requires_official_host_and_hash(self):
        index = "https://download.pytorch.org/whl/cu128/torch/"
        filename = "torch-2.11.0+cu128-cp312-cp312-win_amd64.whl"
        href = "https://download.pytorch.org/whl/cu128/" + filename.replace("+", "%2B")
        result = wheel_from_index(index, f'<a href="{href}#sha256={self.digest}">wheel</a>', filename)
        self.assertEqual(result, (href, filename, self.digest))
        for bad in [href, href.replace("download.pytorch.org", "unrelated.example") + "#sha256=" + self.digest]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                wheel_from_index(index, f'<a href="{bad}">wheel</a>', filename)


if __name__ == "__main__":
    unittest.main()
