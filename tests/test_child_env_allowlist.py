"""Third-party children must not inherit the user's whole credential set.

The engine spawns binaries it does not own: ``npx`` for the Printing Press
CLIs, ``brew`` for yt-dlp, the ``agentcookie`` sidecar, the Bright Data CLI,
and ``openssl``/``security`` for cookie decryption. Every one of them was
started with no ``env=``, so each child inherited ``os.environ`` verbatim --
every API key, the X session cookies, cloud credentials, and SSH agent state.

bird_x (issue #1063) and grok_x already build an allowlisted child env for
exactly this reason; these tests hold the rest of the tree to that standard.
"""

from __future__ import annotations

from unittest import mock

import pytest

from lib import subproc

SECRETS = {
    "OPENAI_API_KEY": "sk-test-not-real",
    "XAI_API_KEY": "xai-test-not-real",
    "GEMINI_API_KEY": "gemini-test-not-real",
    "SCRAPECREATORS_API_KEY": "sc-test-not-real",
    "AUTH_TOKEN": "cookie-test-not-real",
    "CT0": "csrf-test-not-real",
    "AWS_SECRET_ACCESS_KEY": "aws-test-not-real",
    "SSH_AUTH_SOCK": "/tmp/not-a-real-agent.sock",
}


def test_child_env_keeps_platform_vars_and_drops_credentials():
    with mock.patch.dict("os.environ", {**SECRETS, "PATH": "/usr/bin", "HOME": "/tmp"}, clear=False):
        env = subproc.child_env()

    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/tmp"
    for name in SECRETS:
        assert name not in env, f"{name} leaked into the child env"


def test_child_env_carries_only_the_named_secret():
    with mock.patch.dict(
        "os.environ",
        {**SECRETS, "BRIGHTDATA_API_KEY": "bd-test-not-real", "PATH": "/usr/bin"},
        clear=False,
    ):
        env = subproc.child_env(
            extra={"BRIGHTDATA_API_KEY": "bd-test-not-real"},
            prefix_allow=("BRIGHTDATA_",),
        )

    assert env["BRIGHTDATA_API_KEY"] == "bd-test-not-real"
    assert "OPENAI_API_KEY" not in env
    assert "AUTH_TOKEN" not in env


def test_child_env_extra_none_values_are_dropped():
    env = subproc.child_env(extra={"SOMETHING": None})
    assert "SOMETHING" not in env


@pytest.mark.parametrize("prefix", ["BRIGHTDATA_", "AGENTCOOKIE_", "BROWSER_"])
def test_prefix_allow_is_opt_in(prefix):
    name = f"{prefix}SETTING"
    with mock.patch.dict("os.environ", {name: "value", "PATH": "/usr/bin"}, clear=False):
        assert name not in subproc.child_env()
        assert subproc.child_env(prefix_allow=(prefix,))[name] == "value"


def test_npx_install_does_not_inherit_secrets():
    from lib import setup_wizard

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch.dict("os.environ", SECRETS, clear=False), mock.patch.object(
        setup_wizard, "shutil"
    ) as shutil_mod, mock.patch.object(setup_wizard.subprocess, "run", side_effect=fake_run):
        shutil_mod.which.side_effect = lambda name: f"/usr/bin/{name}"
        setup_wizard._run_npx_install("digg-pp-cli")

    env = captured["env"]
    assert env is not None, "install child must get an explicit env"
    for name in SECRETS:
        assert name not in env


def test_agentcookie_does_not_inherit_secrets():
    from lib import agentcookie

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return mock.Mock(
            returncode=0,
            stdout='{"auth_token": "a", "ct0": "b"}',
            stderr="",
        )

    with mock.patch.dict("os.environ", SECRETS, clear=False), mock.patch.object(
        agentcookie.subprocess, "run", side_effect=fake_run
    ), mock.patch.object(
        agentcookie.shutil, "which", return_value="/usr/local/bin/agentcookie"
    ):
        agentcookie.read_x_cookies({"AGENTCOOKIE": "on"})

    env = captured["env"]
    assert env is not None, "sidecar child must get an explicit env"
    for name in SECRETS:
        assert name not in env


def test_brightdata_child_env_never_inherits_secrets():
    from lib import brightdata

    with mock.patch.dict("os.environ", SECRETS, clear=False):
        with_key = brightdata._child_env("bd-test-not-real")
        without_key = brightdata._child_env("")

    for env in (with_key, without_key):
        assert env is not None, "child env must be explicit, never inherited"
        for name in SECRETS:
            assert name not in env
    assert with_key[brightdata.API_KEY_ENV] == "bd-test-not-real"


def test_config_dir_vars_are_forwarded():
    """A CLI that finds its own credentials through XDG_/APPDATA must keep them."""
    with mock.patch.dict(
        "os.environ",
        {"XDG_CONFIG_HOME": "/tmp/xdg", "APPDATA": "C:/appdata", "OPENAI_API_KEY": "sk-test"},
        clear=False,
    ):
        env = subproc.child_env()
    assert env["XDG_CONFIG_HOME"] == "/tmp/xdg"
    assert env["APPDATA"] == "C:/appdata"
    assert "OPENAI_API_KEY" not in env


def test_brightdata_omits_an_empty_key_variable():
    from lib import brightdata

    env = brightdata._child_env("")
    assert brightdata.API_KEY_ENV not in env


def test_keychain_lookup_does_not_inherit_secrets():
    """The child that reads the Safe Storage passphrase gets the allowlist too."""
    from lib import chrome_cookies

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return mock.Mock(returncode=1, stdout="", stderr="")

    with mock.patch.dict("os.environ", SECRETS, clear=False), mock.patch.object(
        chrome_cookies.subprocess, "run", side_effect=fake_run
    ):
        chrome_cookies._get_chromium_encryption_key("Chrome Safe Storage")

    assert captured["cmd"][0] == "security"
    env = captured["env"]
    assert env is not None, "keychain child must get an explicit env"
    for name in SECRETS:
        assert name not in env


def test_cookie_decrypt_does_not_inherit_secrets():
    from lib import chrome_cookies

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return mock.Mock(returncode=1, stdout=b"", stderr=b"")

    with mock.patch.dict("os.environ", SECRETS, clear=False), mock.patch.object(
        chrome_cookies.subprocess, "run", side_effect=fake_run
    ):
        chrome_cookies._decrypt_v10_value(b"v10" + b"\x00" * 16, b"\x11" * 16, 10)

    env = captured["env"]
    assert env is not None, "openssl child must get an explicit env"
    for name in SECRETS:
        assert name not in env
