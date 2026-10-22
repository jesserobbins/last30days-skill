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

import unittest

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
    """

    def serve_redirect(self, location: str) -> str:
        """Start a one-shot server that 302s to ``location``; return its URL."""
        import http.server
        import socketserver
        import threading

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - stdlib name
                self.send_response(302)
                self.send_header("Location", location)
                self.end_headers()

            def log_message(self, *args):
                pass

        socketserver.TCPServer.allow_reuse_address = True
        server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        return f"http://127.0.0.1:{port}/careers"

    def test_ftp_redirect_is_refused(self):
        url = self.serve_redirect("ftp://127.0.0.1:2121/careers")
        with pytest.raises(http.HTTPError):
            http.request("GET", url, raw=True, retries=0)

    def test_file_redirect_is_refused(self):
        url = self.serve_redirect("file:///etc/passwd")
        with pytest.raises(http.HTTPError):
            http.request("GET", url, raw=True, retries=0)

    def test_caller_validator_is_applied_to_the_redirect_target(self):
        # A public careers page that redirects to link-local metadata passes a
        # first-hop-only check.
        url = self.serve_redirect("http://169.254.169.254/latest/meta-data/careers")
        reject_private = lambda candidate: "169.254." not in candidate  # noqa: E731
        with pytest.raises(http.HTTPError):
            http.request(
                "GET", url, raw=True, retries=0, validate_url=reject_private
            )

    def test_http_redirect_still_follows(self):
        target = self.serve_redirect("/ok")
        url = self.serve_redirect(target)
        # No assertion on the body: the point is that an http(s) hop is not
        # refused by the new guard. A connection error here means the one-shot
        # server ran out, which is not what this test is about.
        try:
            http.request("GET", url, raw=True, retries=0)
        except http.HTTPError as error:
            assert "non-HTTP" not in str(error)
