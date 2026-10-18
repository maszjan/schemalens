"""`schemalens-api keys new`: random keys, added to a .env without touching the rest."""

import stat

from schemalens_api.cli import main, new_key


def test_keys_are_random_and_tell_their_kind() -> None:
    assert new_key("API_KEYS") != new_key("API_KEYS")
    assert new_key("API_KEYS").startswith("sl_") and len(new_key("API_KEYS")) > 40
    assert new_key("DEMO_KEYS").startswith("sl_demo_")


def test_a_key_is_printed_once(capsys) -> None:
    assert main(["keys", "new"]) == 0
    (key,) = capsys.readouterr().out.split()
    assert key.startswith("sl_")


def test_keys_are_added_to_the_env_file(tmp_path, capsys) -> None:
    env = tmp_path / ".env"
    env.write_text("APP=schemalens\n# API_KEYS=\nLOCAL_TIMEOUT_S=180\n")

    main(["keys", "new", "--env-file", str(env)])
    first = capsys.readouterr().out.strip()
    main(["keys", "new", "--env-file", str(env)])
    second = capsys.readouterr().out.strip()
    main(["keys", "new", "--demo", "--env-file", str(env)])
    demo = capsys.readouterr().out.strip()

    lines = env.read_text().splitlines()
    assert lines[:3] == ["APP=schemalens", "# API_KEYS=", "LOCAL_TIMEOUT_S=180"]
    assert lines[3:] == [f"API_KEYS={first},{second}", f"DEMO_KEYS={demo}"]
    assert stat.S_IMODE(env.stat().st_mode) == 0o600


def test_an_existing_line_keeps_its_keys(tmp_path, capsys) -> None:
    env = tmp_path / ".env"
    env.write_text('API_KEYS="sl_old"\n')
    main(["keys", "new", "--env-file", str(env)])
    key = capsys.readouterr().out.strip()
    assert env.read_text() == f"API_KEYS=sl_old,{key}\n"
