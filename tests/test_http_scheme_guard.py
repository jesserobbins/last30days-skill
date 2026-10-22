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
