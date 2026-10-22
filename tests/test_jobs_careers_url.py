"""A web-search result must not point the careers probe at the local network.

``_search_for_careers_url`` accepts the first search hit whose text contains a
careers-ish token, then ``jobs.py`` fetches it. Treating a URL from a search
backend as trustworthy makes the probe a request forgery primitive: a poisoned
or SEO'd result aimed at link-local metadata or an internal admin page returns
that page's body, which is then parsed for ATS slugs and rendered into the
report.
"""

from __future__ import annotations

from unittest import mock

import pytest

from lib import jobs


def _search_result(url: str) -> list[dict]:
    return [{"title": "Acme careers", "url": url, "snippet": "open roles"}]


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/careers",
        "http://127.0.0.1:8080/careers",
        "http://[::1]/careers",
        "http://10.0.0.5/jobs",
        "http://192.168.1.10/careers",
        "http://172.16.4.4/jobs",
        "http://localhost/careers",
        "http://metadata.google.internal/jobs",
        "file:///etc/careers",
    ],
)
def test_search_result_cannot_point_at_the_local_network(url):
    with mock.patch.object(jobs.grounding, "web_search", return_value=(_search_result(url), {})):
        assert jobs._search_for_careers_url(
            "Acme", ("2026-01-01", "2026-01-31"), {}, backend="duckduckgo"
        ) is None


def test_public_careers_url_is_still_accepted():
    url = "https://boards.greenhouse.io/acme"
    with mock.patch.object(jobs.grounding, "web_search", return_value=(_search_result(url), {})):
        assert jobs._search_for_careers_url(
            "Acme", ("2026-01-01", "2026-01-31"), {}, backend="duckduckgo"
        ) == url


def test_careers_fetch_skips_a_rejected_url():
    """The resolver must not fall through to fetching a rejected hit."""
    with mock.patch.object(
        jobs.grounding,
        "web_search",
        return_value=(_search_result("http://169.254.169.254/careers"), {}),
    ), mock.patch.object(jobs.http, "get_text", return_value=None) as get_text:
        html, url = jobs._resolve_careers_page(
            "Acme", ("2026-01-01", "2026-01-31"), {}, backend="duckduckgo"
        )
    assert (html, url) == (None, None)
    fetched = [call.args[0] for call in get_text.call_args_list]
    assert not any("169.254.169.254" in candidate for candidate in fetched)
