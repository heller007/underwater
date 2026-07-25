"""Smoke tests for common run scaffolding."""

from __future__ import annotations

from src.common import PROJECT_ROOT, create_run, load_env, resolve_device, set_seed


def test_project_root_exists():
    assert (PROJECT_ROOT / "configs" / "env" / "local.yaml").exists()


def test_create_run(tmp_path):
    set_seed(0)
    ctx = create_run(
        experiment_id="unit",
        seed=0,
        config={"hello": "world"},
        runs_root=tmp_path,
        fold="Lokrum",
        model_tag="t0",
    )
    assert ctx.run_dir.exists()
    assert (ctx.run_dir / "run_manifest.json").exists()
    assert (ctx.run_dir / "config_resolved.yaml").exists()


def test_load_env_local():
    env = load_env("local")
    assert env.name == "local"
    assert env.runs_root.name == "runs"


def test_resolve_device_cpu_or_gpu():
    d = resolve_device("cpu")
    assert d == "cpu"
