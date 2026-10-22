"""Untrusted input must not be able to pick the transport scheme.

``http.request`` handed whatever it was given to ``urllib.request``, and the
opener it builds installs ``FileHandler``, ``FTPHandler``, and ``DataHandler``
alongside the HTTP handlers. A URL that reaches the engine from outside --
a web-search result chosen for the ``--hiring-signals`` careers probe, a
caption URL parsed out of a watch page -- could therefore read a local file or
an internal HTTP endpoint and return the body as if it were fetched HTML.

Only http(s) is a legitimate research transport, so the seam refuses anything
else before a request is built.
"""

from __future__ import annotations

import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

import pytest

from lib import http


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "file:///etc/careers",
        "ftp://example.com/jobs",
        "data:text/html,<h1>careers</h1>",
        "gopher://127.0.0.1:70/jobs",
        "javascript:alert(1)",
    ],
)
def test_non_http_schemes_are_refused(url):
    # raw=True is the shape that returns a body instead of parsing JSON, so the
    # refusal must happen before the transport, not on the decode path.
    with pytest.raises(http.HTTPError):
        http.request("GET", url, raw=True, retries=1)


def test_refusal_does_not_read_the_local_file(tmp_path):
    secret = tmp_path / "careers"
    secret.write_text("TOP-SECRET-LOCAL-CONTENT", encoding="utf-8")

    with pytest.raises(http.HTTPError):
        http.request("GET", f"file://{secret}", retries=0)
    # get_text is the shape jobs.py uses; it must not return the file either.
    # retries=1 is the shipping default path -- with retries=0 the retry loop
    # never runs and returns None for the wrong reason.
    assert http.get_text(f"file://{secret}", accept="text/html", retries=1) is None


def test_scheme_check_is_case_insensitive():
    with pytest.raises(http.HTTPError):
        http.request("GET", "FILE:///etc/passwd", retries=0)


def test_missing_scheme_is_refused():
    with pytest.raises(http.HTTPError):
        http.request("GET", "/etc/passwd", retries=0)


class RedirectSchemeGuardTest(unittest.TestCase):
    """The allowlist must cover every hop, not just the first one.

    CPython's redirect handler accepts an ``ftp://`` target and the shared
    opener then services it with FTPHandler, so a public page answering
    ``302 Location: ftp://internal/`` used to move the engine onto another
    transport with the guard already behind it.

    These tests use retries=1, not 0: ``request`` skips its transport loop
    entirely when the retry budget is zero and raises a generic "Request
    failed with no error details", which would let every assertion below pass
    with the guard deleted.
    """

    _servers: list = []

    def tearDown(self):
        for server in self._servers:
            server.server_close()
        self._servers.clear()

    def _serve(self, body: str | None = None, location: str | None = None) -> str:
        """Serve ``body`` with 200, or redirect to ``location``; return the URL."""
        import http.server
        import socketserver

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - stdlib name
                if location is not None:
                    self.send_response(302)
                    self.send_header("Location", location)
                    self.end_headers()
                    return
                payload = (body or "").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        socketserver.TCPServer.allow_reuse_address = True
        server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        self._servers.append(server)
        handler = lambda: server.handle_request()  # noqa: E731
        threading.Thread(target=handler, daemon=True).start()
        return f"http://127.0.0.1:{server.server_address[1]}/careers"

    def test_ftp_redirect_is_refused(self):
        url = self._serve(location="ftp://127.0.0.1:2121/careers")
        with pytest.raises(http.HTTPError) as error:
            http.request("GET", url, raw=True, retries=1, timeout=5)
        assert "non-HTTP(S)" in str(error.value)

    def test_file_redirect_is_refused(self):
        # CPython refuses file:// redirects itself (its own allowlist is
        # http/https/ftp); this pins that the local file is never read.
        url = self._serve(location="file:///tmp/last30days-should-not-be-read")
        with pytest.raises(http.HTTPError) as error:
            http.request("GET", url, raw=True, retries=1, timeout=5)
        assert "not allowed" in str(error.value) or "non-HTTP(S)" in str(error.value)

    def test_caller_validator_is_applied_to_the_redirect_target(self):
        # A public careers page redirecting to link-local metadata passes a
        # first-hop-only check, so the validator has to ride the redirect.
        # Tested on the handler directly: a request carrying a validator is
        # also pinned to a public address, so a loopback test server would be
        # refused before the redirect is ever seen.
        handler = http._StripAuthOnCrossOriginRedirect()
        request = urllib.request.Request("http://careers.example/jobs")
        request.last30days_url_validator = (
            lambda candidate: "169.254." not in candidate
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            handler.redirect_request(
                request,
                None,
                302,
                "Found",
                {},
                "http://169.254.169.254/latest/meta-data/careers",
            )
        assert "refused URL" in str(error.value)

    def test_validator_rides_the_redirect_to_the_next_hop(self):
        handler = http._StripAuthOnCrossOriginRedirect()
        request = urllib.request.Request("http://careers.example/jobs")
        validator = lambda candidate: True  # noqa: E731
        request.last30days_url_validator = validator
        new = handler.redirect_request(
            request, None, 302, "Found", {}, "http://careers.example/roles"
        )
        assert new.last30days_url_validator is validator

    def test_http_redirect_is_followed_to_the_body(self):
        terminal = self._serve(body="<html>public careers page</html>")
        url = self._serve(location=terminal)
        body = http.request("GET", url, raw=True, retries=1, timeout=5)
        assert "public careers page" in body


class PinnedAddressTest(unittest.TestCase):
    """Validating the name must not leave the connection address unchecked.

    The lookup that clears a URL and the lookup http.client performs when it
    connects are separate, so an attacker-controlled zone can answer publicly
    first and with a loopback address for the real connection.
    """

    def test_connection_refuses_a_non_public_answer(self):
        """Second lookup returning 127.0.0.1 must not be dialled."""
        import socket

        connection = http._PinnedHTTPSConnection("careers.attacker.example", timeout=5)
        dialled: list = []

        def rebound(host, *args, **kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
            ]

        connection._create_connection = lambda *a, **k: dialled.append(a) or mock.MagicMock()
        with mock.patch.object(http.socket, "getaddrinfo", side_effect=rebound):
            with pytest.raises(urllib.error.URLError):
                connection.connect()
        assert dialled == [], "a loopback address was dialled"

    def test_connection_dials_the_public_answer(self):
        import socket

        connection = http._PinnedHTTPSConnection("careers.example", timeout=5)
        dialled: list = []
        connection._create_connection = lambda *a, **k: dialled.append(a) or mock.MagicMock()

        with mock.patch.object(
            http.socket, "getaddrinfo",
            return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 443)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
            ],
        ):
            connection.connect()

        assert dialled == [(("93.184.216.34", 443), 5, None)], dialled

    def test_hostname_is_preserved_for_the_host_header(self):
        connection = http._PinnedHTTPSConnection("careers.example", timeout=5)
        assert connection.host == "careers.example"
