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
        # Carrier-grade NAT (100.64.0.0/10) holds Alibaba Cloud's metadata
        # endpoint; ipaddress.is_private is False for it.
        "http://100.100.100.200/latest/meta-data/careers",
        # Scoped literals: ipaddress cannot parse a scope id, so these used to
        # fall through to the resolver branch and be accepted.
        "http://[fe80::1%25eth0]/careers",
        "http://[::1%251]/careers",
        # Non-canonical numeric spellings that ipaddress rejects but the
        # resolver still honours.
        "http://127.1/jobs",
        "http://2130706433/jobs",
        "file:///etc/careers",
    ],
)
def test_search_result_cannot_point_at_the_local_network(url):
    with mock.patch.object(jobs.grounding, "web_search", return_value=(_search_result(url), {})):
        assert jobs._search_for_careers_url(
            "Acme", ("2026-01-01", "2026-01-31"), {}, backend="duckduckgo"
        ) is None


def _allow_dns(monkeypatch, address: str = "8.8.8.8"):
    """Stub resolution so the suite's network guard stays closed.

    The default is a globally reachable address: documentation ranges such as
    203.0.113.0/24 are reserved, so is_global is False for them and they would
    be rejected on the positive paths.
    """
    import ipaddress
    import socket as socket_mod

    def fake_getaddrinfo(host, *args, **kwargs):
        return [(socket_mod.AF_INET, socket_mod.SOCK_STREAM, 6, "", (address, 0))]

    monkeypatch.setattr(jobs.socket, "getaddrinfo", fake_getaddrinfo)
    return ipaddress.ip_address(address)


def test_hostname_resolving_to_loopback_is_rejected(monkeypatch):
    """A DNS record pointing at 127.0.0.1 needs no rebinding to be useful."""
    _allow_dns(monkeypatch, "127.0.0.1")
    with mock.patch.object(
        jobs.grounding, "web_search", return_value=(_search_result("http://careers.attacker.example/jobs"), {})
    ):
        assert jobs._search_for_careers_url(
            "Acme", ("2026-01-01", "2026-01-31"), {}, backend="duckduckgo"
        ) is None


def test_hostname_resolving_to_metadata_is_rejected(monkeypatch):
    _allow_dns(monkeypatch, "169.254.169.254")
    assert jobs._is_fetchable_url("http://careers.attacker.example/jobs") is False


def test_unresolvable_hostname_does_not_block_the_fetch(monkeypatch):
    """Resolution failure must not change the predicate: the fetch fails anyway."""
    import socket as socket_mod

    def broken(host, *args, **kwargs):
        raise socket_mod.gaierror("no dns here")

    monkeypatch.setattr(jobs.socket, "getaddrinfo", broken)
    assert jobs._is_fetchable_url("http://careers.example/jobs") is True


def test_empty_host_returns_false():
    assert jobs._is_fetchable_url("http:///jobs") is False


def test_invalid_label_is_rejected(monkeypatch):
    """An invalid label can never be fetched, so it is refused, not waved through."""
    monkeypatch.setattr(
        jobs.socket, "getaddrinfo", mock.Mock(side_effect=UnicodeError("label too long"))
    )
    assert jobs._is_fetchable_url("http://" + "a" * 64 + ".com/jobs") is False


def test_transient_resolution_failure_fails_open(monkeypatch):
    monkeypatch.setattr(jobs.socket, "getaddrinfo", mock.Mock(side_effect=OSError("nope")))
    assert jobs._is_fetchable_url("http://careers.example/jobs") is True


def test_long_company_name_skips_the_candidate_without_raising(monkeypatch):
    """A >63-char label must not raise out of a documented never-raises path."""
    monkeypatch.setattr(
        jobs.socket, "getaddrinfo", mock.Mock(side_effect=UnicodeError("label too long"))
    )
    with mock.patch.object(jobs.http, "get_text", return_value=None) as get_text:
        assert jobs._resolve_careers_page(
            "a" * 70, ("2026-01-01", "2026-01-31"), {}, backend="none"
        ) == (None, None)
    assert get_text.call_count >= 0  # never raised


def test_unparsable_resolved_address_is_rejected(monkeypatch):
    """An address we cannot classify is not one we fetch (fail closed)."""
    import socket as socket_mod

    monkeypatch.setattr(
        jobs.socket,
        "getaddrinfo",
        lambda host, *a, **k: [(socket_mod.AF_INET6, socket_mod.SOCK_STREAM, 6, "", ("fe80::1%eth0", 0, 0, 0))],
    )
    assert jobs._is_fetchable_url("http://careers.example/jobs") is False


def test_non_careers_results_cost_no_dns_lookup(monkeypatch):
    """The address check resolves hostnames, so it must run after the token filter."""
    calls: list[str] = []

    def counting_getaddrinfo(host, *args, **kwargs):
        calls.append(host)
        raise OSError("resolution should not be attempted for this result")

    monkeypatch.setattr(jobs.socket, "getaddrinfo", counting_getaddrinfo)
    results = [
        {"title": "unrelated", "url": "https://news.example/politics"},
        {"title": "unrelated 2", "url": "https://blog.example/post"},
    ]
    with mock.patch.object(jobs.grounding, "web_search", return_value=(results, {})):
        assert jobs._search_for_careers_url(
            "Acme", ("2026-01-01", "2026-01-31"), {}, backend="duckduckgo"
        ) is None
    assert calls == []


def test_malformed_search_hit_does_not_abort_the_probe(monkeypatch):
    """One unparsable hit must be skipped, not raise out of discovery."""
    _allow_dns(monkeypatch)
    results = [
        {"title": "broken", "url": "http://[broken/about", "snippet": "careers"},
        {"title": "Acme careers", "url": "https://boards.greenhouse.io/acme"},
    ]
    with mock.patch.object(jobs.grounding, "web_search", return_value=(results, {})):
        assert jobs._search_for_careers_url(
            "Acme", ("2026-01-01", "2026-01-31"), {}, backend="duckduckgo"
        ) == "https://boards.greenhouse.io/acme"


def test_public_careers_url_is_still_accepted(monkeypatch):
    _allow_dns(monkeypatch)
    url = "https://boards.greenhouse.io/acme"
    with mock.patch.object(jobs.grounding, "web_search", return_value=(_search_result(url), {})):
        assert jobs._search_for_careers_url(
            "Acme", ("2026-01-01", "2026-01-31"), {}, backend="duckduckgo"
        ) == url


def test_public_fetch_survives_a_locally_resolving_guessed_domain(monkeypatch):
    """The guessed {slug}.com candidates get the same per-hop rule."""
    _allow_dns(monkeypatch, "127.0.0.1")
    with mock.patch.object(jobs.http, "get_text", return_value=None) as get_text:
        jobs._resolve_careers_page(
            "Acme", ("2026-01-01", "2026-01-31"), {}, backend="none"
        )
    for call in get_text.call_args_list:
        assert call.kwargs.get("validate_url") is jobs._is_fetchable_url


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
