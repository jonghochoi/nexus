#!/usr/bin/env python3
"""
tests/smoke_test.py — NEXUS local validation script
=====================================================
Automated validation script that runs identically on both local PC and GPU server.

Prerequisites:
  - nexus venv activated (source ~/.nexus/activate.sh, or `nexus-activate`)
  - MLflow server running (bash scheduled_sync/start_local_mlflow.sh)

Usage:
  python tests/smoke_test.py
  python tests/smoke_test.py --tracking-uri http://127.0.0.1:5100
  python tests/smoke_test.py --tracking-uri http://<nexus-server>:5000
"""

import argparse
import sys
import time

# ── ANSI colors ──────────────────────────────────────────────────────────────
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"

PASS = f"{GREEN}[PASS]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
INFO = f"{CYAN}[INFO]{RESET}"
WARN = f"{YELLOW}[WARN]{RESET}"


def section(title: str) -> None:
    print(f"\n{BOLD}{'─' * 55}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * 55}{RESET}")


def ok(msg: str) -> None:
    print(f"  {PASS}  {msg}")


def fail(msg: str) -> None:
    print(f"  {FAIL}  {msg}")


def info(msg: str) -> None:
    print(f"  {INFO}  {msg}")


# ── Test functions ───────────────────────────────────────────────────────────


def check_imports() -> bool:
    """1. Check required package imports"""
    section("1. Package Import Check")
    packages = {
        "mlflow": "mlflow",
        "tbparse": "tbparse",
        "tensorboard": "tensorboard",
        "tensorboardX": "tensorboardX",
        "pandas": "pandas",
        "rich": "rich",
    }
    all_ok = True
    for name, module in packages.items():
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "?")
            ok(f"{name} ({version})")
        except ImportError as e:
            fail(f"{name} — {e}")
            all_ok = False

    # logger package (nexus internal module) — core
    try:
        sys.path.insert(0, ".")
        from nexus.logger import make_logger, MLflowLogger, DualLogger, TBLogger

        ok("nexus.logger core — make_logger, MLflowLogger, DualLogger, TBLogger")
    except ImportError as e:
        fail(f"nexus.logger core — {e}")
        all_ok = False

    # logger advanced features — explicit import paths
    try:
        from nexus.logger.sweep_logger import SweepLogger  # noqa: F401
        from nexus.logger.model_registry import ModelRegistry  # noqa: F401
        from nexus.logger.system_metrics import SystemMetricsLogger  # noqa: F401
        from nexus.logger.eval_logger import EvalLogger  # noqa: F401

        ok("nexus.logger advanced — SweepLogger, ModelRegistry, SystemMetricsLogger, EvalLogger")
    except ImportError as e:
        fail(f"nexus.logger advanced — {e}")
        all_ok = False

    return all_ok


def check_mlflow_connection(tracking_uri: str) -> bool:
    """2. Check MLflow server connection"""
    section("2. MLflow Server Connection Check")
    info(f"URI: {tracking_uri}")
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        # verify connection by listing experiments
        experiments = client.search_experiments()
        ok(f"MLflow server connected ({len(experiments)} experiments found)")
        return True
    except Exception as e:
        fail(f"MLflow server connection failed: {e}")
        print(f"\n  {WARN}  Check that the MLflow server is running:")
        print(f"         bash scheduled_sync/start_local_mlflow.sh")
        return False


def test_mlflow_logger(tracking_uri: str) -> bool:
    """3. MLflowLogger logging test"""
    section("3. MLflowLogger Logging Test")
    try:
        sys.path.insert(0, ".")
        from nexus.logger import MLflowLogger

        run_name = f"smoke_test_{int(time.time())}"
        info(f"Creating test run: {run_name}")

        logger = MLflowLogger(
            run_name=run_name,
            tracking_uri=tracking_uri,
            experiment_name="nexus_smoke_test",
            params={"lr": 0.001, "batch_size": 64, "nested": {"gamma": 0.99}},
            tags={"researcher": "smoke_test", "task": "verification"},
        )

        # log dummy metrics (3 steps)
        for step in range(1, 4):
            logger.add_scalar("train/reward", float(step * 10), step)
            logger.add_scalar("train/loss", 1.0 / step, step)
            logger.add_scalar("eval/success_rate", step * 0.3, step)
        logger.close()

        ok("MLflowLogger logging complete")

        # verify logged data
        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        exp = mlflow.set_tracking_uri(tracking_uri) or mlflow.get_experiment_by_name(
            "nexus_smoke_test"
        )

        import mlflow as _mlflow

        _mlflow.set_tracking_uri(tracking_uri)
        exp = _mlflow.get_experiment_by_name("nexus_smoke_test")

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'"
        )
        if not runs:
            fail("Recorded run not found")
            return False

        run_data = runs[0]
        metrics = run_data.data.metrics
        expected_keys = {"train/reward", "train/loss", "eval/success_rate"}
        found_keys = set(metrics.keys())

        if not expected_keys.issubset(found_keys):
            missing = expected_keys - found_keys
            fail(f"Missing metrics: {missing}")
            return False

        ok(f"Metrics recorded: {sorted(found_keys)}")

        # validate last step values
        assert abs(metrics["train/reward"] - 30.0) < 1e-6, "train/reward value mismatch"
        assert abs(metrics["eval/success_rate"] - 0.9) < 1e-6, "eval/success_rate value mismatch"
        ok("Metric value accuracy validated")

        # validate hyperparameters
        params = run_data.data.params
        if "lr" in params and "batch_size" in params:
            ok(f"Hyperparameters recorded: lr={params['lr']}, batch_size={params['batch_size']}")
        else:
            fail(f"Hyperparameter logging failed. Found params: {list(params.keys())}")
            return False

        return True

    except Exception as e:
        fail(f"MLflowLogger test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_split_params(tracking_uri: str) -> bool:
    """3b. agent_params / env_params — prefixed MLflow params + JSON artifacts"""
    section("3b. Split Params (agent_params / env_params)")
    try:
        sys.path.insert(0, ".")
        from nexus.logger import MLflowLogger

        run_name = f"split_params_test_{int(time.time())}"
        agent_cfg = {"lr": 3e-4, "clip_eps": 0.2, "network": {"hidden": 256}}
        # Include a class ref and a callable to exercise _to_jsonable —
        # IsaacLab/Hydra env configs routinely embed `class_type` / `func` as values,
        # and the JSON artifact build must reduce them to qualified names rather
        # than handing raw `type` objects to json.dump.

        class _DummyAction:
            pass

        def _dummy_reward(env):
            return 0.0

        env_cfg = {
            "task": "robot_hand",
            "max_steps": 1000,
            "reward_scale": 0.1,
            "class_type": _DummyAction,
            "reward_fn": _dummy_reward,
        }

        logger = MLflowLogger(
            run_name=run_name,
            tracking_uri=tracking_uri,
            experiment_name="nexus_smoke_test",
            agent_params=agent_cfg,
            env_params=env_cfg,
            tags={"researcher": "smoke_test"},
        )
        logger.close()

        import mlflow as _mlflow
        from mlflow.tracking import MlflowClient

        _mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        exp = _mlflow.get_experiment_by_name("nexus_smoke_test")
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'"
        )
        if not runs:
            fail("Split-params run not found")
            return False

        # ── validate prefixed MLflow params ──────────────────────────────────
        params = runs[0].data.params
        expected_keys = {
            "agent.lr",
            "agent.clip_eps",
            "agent.network.hidden",
            "env.task",
            "env.max_steps",
            "env.reward_scale",
        }
        missing = expected_keys - set(params.keys())
        if missing:
            fail(f"Prefixed params missing: {missing}")
            return False
        ok(f"Prefixed MLflow params present: {sorted(expected_keys)}")

        # ── validate JSON artifacts ───────────────────────────────────────────
        run_id = runs[0].info.run_id
        artifacts = {a.path for a in client.list_artifacts(run_id, "params")}
        expected_artifacts = {"params/agent_params.json", "params/env_params.json"}
        missing_artifacts = expected_artifacts - artifacts
        if missing_artifacts:
            fail(f"Param artifacts missing: {missing_artifacts}")
            return False
        ok(f"Param artifacts present: params/agent_params.json, params/env_params.json")

        return True

    except Exception as e:
        fail(f"Split params test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_make_logger_factory(tracking_uri: str) -> bool:
    """4. make_logger factory (mode='mlflow') test"""
    section("4. make_logger Factory Test")
    try:
        sys.path.insert(0, ".")
        from nexus.logger import make_logger

        run_name = f"factory_test_{int(time.time())}"
        logger = make_logger(
            mode="mlflow",
            run_name=run_name,
            tracking_uri=tracking_uri,
            experiment_name="nexus_smoke_test",
            params={"env": "smoke"},
            tags={"researcher": "smoke_test"},
        )
        logger.add_scalar("test/metric", 1.0, 1)
        logger.add_scalar("test/metric", 2.0, 2)
        logger.close()
        ok(f"make_logger(mode='mlflow') working correctly")
        return True
    except Exception as e:
        fail(f"make_logger test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_dual_logger(tracking_uri: str) -> bool:
    """5. DualLogger (mode='dual') test"""
    section("5. DualLogger Test (TensorBoard + MLflow)")
    try:
        import tempfile, os

        sys.path.insert(0, ".")
        from nexus.logger import make_logger

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_name = f"dual_test_{int(time.time())}"
            logger = make_logger(
                mode="dual",
                tb_dir=tmp_dir,
                run_name=run_name,
                tracking_uri=tracking_uri,
                experiment_name="nexus_smoke_test",
                params={"mode": "dual"},
                tags={"researcher": "smoke_test"},
            )
            for step in range(1, 6):
                logger.add_scalar("dual/reward", float(step * 5), step)
            logger.close()

            # check tfevents file creation
            tb_files = [f for f in os.listdir(tmp_dir) if "tfevents" in f]
            if tb_files:
                ok(f"TensorBoard tfevents file created: {tb_files[0]}")
            else:
                fail("tfevents file was not created")
                return False

        ok("DualLogger working correctly (TensorBoard + MLflow simultaneous logging)")
        return True
    except Exception as e:
        fail(f"DualLogger test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


# ── Advanced tests ───────────────────────────────────────────────────────────


def test_omegaconf_flatten(tracking_uri: str) -> bool:
    """6. _flatten() handles OmegaConf DictConfig"""
    section("6. OmegaConf DictConfig Flatten")
    try:
        try:
            from omegaconf import OmegaConf
        except ImportError:
            print(f"  {WARN}  omegaconf not installed - skipping (pip install omegaconf to enable)")
            return True

        sys.path.insert(0, ".")
        from nexus.logger import MLflowLogger

        cfg = OmegaConf.create(
            {
                "lr": 1e-3,
                "trainer": {"clip_eps": 0.2, "gamma": 0.99},
                "env": {"name": "robot_hand", "physics": {"solver": "TGS"}},
            }
        )

        run_name = f"omegaconf_test_{int(time.time())}"
        logger = MLflowLogger(
            run_name=run_name,
            tracking_uri=tracking_uri,
            experiment_name="nexus_smoke_test",
            params=cfg,
        )
        logger.close()

        import mlflow as _mlflow
        from mlflow.tracking import MlflowClient

        _mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        exp = _mlflow.get_experiment_by_name("nexus_smoke_test")
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'"
        )
        if not runs:
            fail("OmegaConf test run not found")
            return False
        params = runs[0].data.params
        expected = {"lr", "trainer.clip_eps", "trainer.gamma", "env.name", "env.physics.solver"}
        missing = expected - set(params.keys())
        if missing:
            fail(f"DictConfig was not flattened: missing keys {missing}")
            return False
        ok(f"DictConfig flattened correctly: {sorted(expected)}")
        return True
    except Exception as e:
        fail(f"OmegaConf flatten test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_scheduled_sync_roundtrip(tracking_uri: str) -> bool:
    """8. scheduled_sync: export_delta -> import_delta round-trip"""
    section("8. scheduled_sync round-trip (export_delta -> import_delta)")
    # Validates the same code that runs under cron, end-to-end against a single
    # MLflow server: log a run in a "source" experiment, export a delta JSON,
    # rewrite the experiment name in-place to a "destination" experiment, import
    # the delta, then verify metrics and the new sync-metadata tags landed.
    # Skips the SCP/SSH transport — that is operator infrastructure, not code
    # under test — but exercises the JSON contract that ties the two scripts.
    try:
        import json as _json
        import os
        import subprocess
        import tempfile
        from pathlib import Path

        sys.path.insert(0, ".")
        from nexus.logger import MLflowLogger

        ts = int(time.time())
        src_exp = f"nexus_smoke_sync_src_{ts}"
        dst_exp = f"nexus_smoke_sync_dst_{ts}"
        run_name = f"sync_roundtrip_{ts}"

        repo_root = Path(__file__).resolve().parent.parent
        export_py = repo_root / "scheduled_sync" / "export_delta.py"
        import_py = repo_root / "scheduled_sync" / "import_delta.py"

        # ── 1. Create a source run with deterministic metrics
        logger = MLflowLogger(
            run_name=run_name,
            tracking_uri=tracking_uri,
            experiment_name=src_exp,
            params={"lr": 0.0007, "algo": "ppo"},
            tags={"researcher": "smoke_test", "task": "sync_roundtrip"},
        )
        for step in range(1, 6):
            logger.add_scalar("train/reward", float(step * 7), step)
            logger.add_scalar("train/loss", 1.0 / step, step)
        logger.close()
        ok(f"Source run logged in {src_exp}")

        # ── 2. export_delta.py — keep state + delta inside a temp dir
        with tempfile.TemporaryDirectory() as tmp:
            # export_delta.py writes a tar.gz bundle (delta.json + artifacts/).
            delta_path = Path(tmp) / "delta.tar.gz"
            state_path = Path(tmp) / "state.json"

            r = subprocess.run(
                [
                    "python",
                    str(export_py),
                    "--tracking-uri",
                    tracking_uri,
                    "--experiment",
                    src_exp,
                    "--output",
                    str(delta_path),
                    "--state-file",
                    str(state_path),
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                fail(f"export_delta.py failed (exit {r.returncode})")
                print(r.stdout)
                print(r.stderr)
                return False
            if not delta_path.exists():
                fail("export_delta.py produced no delta file")
                return False
            ok(f"export_delta.py wrote delta ({delta_path.stat().st_size} bytes)")

            # ── 3. Rewrite the delta to target a fresh experiment so import_delta
            #      creates new runs (instead of resuming the source ones). The
            #      bundle is a tar.gz of `delta.json` + `artifacts/<run_id>/...`,
            #      so we extract, patch the experiment field, and repack.
            import tarfile as _tarfile

            extract_dir = Path(tmp) / "extracted"
            extract_dir.mkdir()
            with _tarfile.open(delta_path, "r:gz") as tar:
                tar.extractall(extract_dir)

            inner_json = extract_dir / "delta.json"
            with open(inner_json) as f:
                delta = _json.load(f)
            assert delta.get("source_host"), "source_host missing from delta"
            delta["experiment"] = dst_exp
            with open(inner_json, "w") as f:
                _json.dump(delta, f)

            # Repack with the same layout (delta.json + optional artifacts/).
            delta_path.unlink()
            with _tarfile.open(delta_path, "w:gz") as tar:
                tar.add(inner_json, arcname="delta.json")
                artifacts_dir = extract_dir / "artifacts"
                if artifacts_dir.exists():
                    tar.add(artifacts_dir, arcname="artifacts")

            # ── 4. import_delta.py — same tracking server, fresh experiment
            r = subprocess.run(
                [
                    "python",
                    str(import_py),
                    "--delta-file",
                    str(delta_path),
                    "--central-tracking-uri",
                    tracking_uri,
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                fail(f"import_delta.py failed (exit {r.returncode})")
                print(r.stdout)
                print(r.stderr)
                return False
            ok("import_delta.py completed")

        # ── 5. Verify destination run carries metrics + sync metadata tags
        import mlflow as _mlflow
        from mlflow.tracking import MlflowClient

        _mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        exp = _mlflow.get_experiment_by_name(dst_exp)
        if exp is None:
            fail(f"Destination experiment {dst_exp} was not created on import")
            return False

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'"
        )
        if not runs:
            fail(f"Imported run {run_name} not found in {dst_exp}")
            return False
        imported = runs[0]

        # Last step value should match what the logger recorded (reward 5*7 = 35)
        if abs(imported.data.metrics.get("train/reward", -1) - 35.0) > 1e-6:
            fail(f"Imported train/reward mismatch: {imported.data.metrics.get('train/reward')!r}")
            return False
        ok("Imported metrics match source values")

        if "lr" not in imported.data.params or "algo" not in imported.data.params:
            fail(f"Imported params missing: {list(imported.data.params.keys())}")
            return False
        ok("Imported params present")

        tags = imported.data.tags
        if not tags.get("nexus.lastSyncTime"):
            fail("nexus.lastSyncTime tag was not stamped on import")
            return False
        if not tags.get("nexus.syncedFromHost"):
            fail("nexus.syncedFromHost tag was not stamped on import")
            return False
        ok(
            f"Sync metadata tags present (lastSyncTime={tags['nexus.lastSyncTime']}, "
            f"syncedFromHost={tags['nexus.syncedFromHost']})"
        )
        return True

    except Exception as e:
        fail(f"sync round-trip test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sweep_logger(tracking_uri: str) -> bool:
    """7. SweepLogger parent-child run test"""
    section("7. SweepLogger (Parent-Child Runs)")
    try:
        sys.path.insert(0, ".")
        from nexus.logger.sweep_logger import SweepLogger
        from nexus.logger import MLflowLogger
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)

        # ── 11a. Context manager — happy path ────────────────────────────────
        sweep_name = f"smoke_sweep_{int(time.time())}"
        with SweepLogger(
            sweep_name=sweep_name,
            tracking_uri=tracking_uri,
            experiment_name="nexus_smoke_test",
            sweep_params={"lr_range": "[1e-4, 1e-3]", "n_trials": "3"},
        ) as sweep:
            parent_id = sweep.parent_run_id
            ok(f"Parent run created: {parent_id[:8]}...")

            for i, lr in enumerate([1e-4, 1e-3], start=1):
                child = MLflowLogger(
                    run_name=f"{sweep_name}_trial_{i}",
                    tracking_uri=tracking_uri,
                    experiment_name="nexus_smoke_test",
                    params={"lr": lr},
                    parent_run_id=parent_id,
                )
                child.add_scalar("train/reward", float(i * 10), 1)
                child.close()
            ok("Child runs created with parent_run_id tag")

            sweep.log_summary(best_params={"lr": 1e-4}, best_metrics={"reward": 20.0})

        # Verify parent run is FINISHED
        finished_run = client.get_run(parent_id)
        if finished_run.info.status != "FINISHED":
            fail(f"Expected FINISHED after clean exit, got {finished_run.info.status!r}")
            return False
        ok("Parent run marked FINISHED after clean context manager exit")

        # Verify parentRunId tag on child run
        exp_obj = client.get_experiment_by_name("nexus_smoke_test")
        child_runs = client.search_runs(
            experiment_ids=[exp_obj.experiment_id],
            filter_string=f"tags.mlflow.runName = '{sweep_name}_trial_1'",
        )
        child_tags = child_runs[0].data.tags
        if child_tags.get("mlflow.parentRunId") != parent_id:
            fail(f"parentRunId tag mismatch: {child_tags.get('mlflow.parentRunId')!r}")
            return False
        ok("mlflow.parentRunId tag verified on child run")

        # ── 11b. Context manager — exception marks run FAILED ────────────────
        failed_name = f"smoke_sweep_fail_{int(time.time())}"
        failed_id = None
        try:
            with SweepLogger(
                sweep_name=failed_name,
                tracking_uri=tracking_uri,
                experiment_name="nexus_smoke_test",
            ) as sweep_fail:
                failed_id = sweep_fail.parent_run_id
                raise RuntimeError("simulated sweep crash")
        except RuntimeError:
            pass  # expected

        if failed_id is not None:
            failed_run = client.get_run(failed_id)
            if failed_run.info.status != "FAILED":
                fail(f"Expected FAILED after exception, got {failed_run.info.status!r}")
                return False
            ok("Parent run marked FAILED after exception in context manager")

        return True
    except Exception as e:
        fail(f"SweepLogger test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_eval_logger(tracking_uri: str) -> bool:
    """9. EvalLogger — attach eval artifacts to an existing run"""
    section("9. EvalLogger (Eval Artifact Upload)")
    try:
        import json
        import tempfile
        from pathlib import Path

        from nexus.logger.eval_logger import EvalLogger
        from nexus.logger import MLflowLogger
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        run_name = f"smoke_eval_{int(time.time())}"

        # ── 9a. Create a target run ───────────────────────────────────────────
        parent = MLflowLogger(
            run_name=run_name,
            tracking_uri=tracking_uri,
            experiment_name="nexus_smoke_test",
            tags={"researcher": "smoke_test"},
        )
        parent.add_scalar("train/reward", 42.0, 1)
        parent.close()
        ok(f"Target run created: {run_name}")

        # ── 9b. Build a minimal eval_dir in a tempdir ─────────────────────────
        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = Path(tmp) / "eval_out"
            eval_dir.mkdir()
            (eval_dir / "report.md").write_text("# Smoke test eval\nAll good.", encoding="utf-8")
            (eval_dir / "metrics.json").write_text(
                json.dumps({"success_rate": 0.87, "nested": {"speed": 1.2}}), encoding="utf-8"
            )
            # Placeholder .mp4s — content isn't a valid video, but their
            # presence exercises the auto-generated index.html's <video>
            # embed path. A root-level video plus a videos/ subdirectory
            # with two more covers the multi-video subdirectory grouping.
            (eval_dir / "rollout.mp4").write_bytes(b"\x00")
            vids = eval_dir / "videos"
            vids.mkdir()
            (vids / "front.mp4").write_bytes(b"\x00")
            (vids / "side.mp4").write_bytes(b"\x00")

            # ── 9c. Upload via EvalLogger ─────────────────────────────────────
            ev = EvalLogger(
                run_name=run_name,
                tracking_uri=tracking_uri,
                experiment=("nexus_smoke_test"),
                verbose=False,
            )
            eval_id = ev.upload(
                eval_dir=eval_dir,
                eval_id="smoke",
                metrics={"explicit_metric": 0.99},
                metrics_from=eval_dir / "metrics.json",
                tags={"observer": "smoke"},
                generate_index=True,
            )

        if eval_id != "smoke":
            fail(f"Expected eval_id='smoke', got {eval_id!r}")
            return False
        ok("upload() returned correct eval_id")

        # ── 9d. Verify tags and metrics were set on the run ───────────────────
        exp_obj = client.get_experiment_by_name("nexus_smoke_test")
        runs = client.search_runs(
            experiment_ids=[exp_obj.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
        )
        run_data = runs[0].data

        if run_data.tags.get("eval.last_id") != "smoke":
            fail(f"eval.last_id tag not set correctly: {run_data.tags.get('eval.last_id')!r}")
            return False
        ok("eval.last_id tag verified")

        if run_data.tags.get("eval.observer") != "smoke":
            fail(f"eval.observer tag not set: {run_data.tags.get('eval.observer')!r}")
            return False
        ok("eval.observer tag verified")

        metric_keys = {m for m in run_data.metrics}
        expected_metrics = {"eval/explicit_metric", "eval/success_rate", "eval/nested.speed"}
        missing = expected_metrics - metric_keys
        if missing:
            fail(f"Missing metrics on run: {missing}")
            return False
        ok(f"Metrics verified: {sorted(expected_metrics)}")

        # ── 9e. Verify auto index.html embeds the placeholder mp4 ─────────────
        artifacts = {a.path for a in client.list_artifacts(runs[0].info.run_id, "eval/smoke")}
        expected_artifacts = {
            "eval/smoke/rollout.mp4",
            "eval/smoke/index.html",
            "eval/smoke/report.md",
            "eval/smoke/metrics.json",
        }
        missing_artifacts = expected_artifacts - artifacts
        if missing_artifacts:
            fail(f"Missing artifacts under eval/smoke/: {missing_artifacts}")
            return False
        ok("eval/smoke/ contains video, index.html, and as-is files")

        with tempfile.TemporaryDirectory() as dl:
            local_index = client.download_artifacts(
                runs[0].info.run_id, "eval/smoke/index.html", dl
            )
            index_html = Path(local_index).read_text(encoding="utf-8")
        if "<video" not in index_html:
            fail("Auto-generated index.html is missing the <video> element")
            return False
        if 'src="data:video/mp4;base64,' not in index_html:
            fail("Auto-generated index.html does not embed mp4 as a data: URI")
            return False
        if "rollout.mp4" not in index_html:
            fail("Auto-generated index.html does not label rollout.mp4")
            return False
        ok("Auto-generated index.html embeds rollout.mp4 inline as data:video/mp4 URI")

        # ── 9f. Verify multi-video subdirectory grouping ──────────────────────
        if index_html.count("<video") < 3:
            fail(f"index.html should embed 3 videos, found {index_html.count('<video')}")
            return False
        for needle in ("<h2>Rollouts</h2>", "<h2>videos/</h2>"):
            if needle not in index_html:
                fail(f"index.html missing expected section header: {needle!r}")
                return False
        for label in ("videos/front.mp4", "videos/side.mp4"):
            if label not in index_html:
                fail(f"index.html does not label {label}")
                return False
        subdir_artifacts = {
            a.path for a in client.list_artifacts(runs[0].info.run_id, "eval/smoke/videos")
        }
        expected_subdir = {"eval/smoke/videos/front.mp4", "eval/smoke/videos/side.mp4"}
        if expected_subdir - subdir_artifacts:
            fail(f"Missing subdir video artifacts: {expected_subdir - subdir_artifacts}")
            return False
        ok("Multi-video subdirectory grouping: Rollouts + videos/ sections verified")

        return True
    except Exception as e:
        fail(f"EvalLogger test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_post_register(tracking_uri: str) -> bool:
    """10. ModelRegistry.register_from_run_name — post-hoc registration"""
    section("10. Post-hoc Model Registry registration")
    try:
        import tempfile
        from pathlib import Path

        from mlflow.tracking import MlflowClient
        from nexus.logger import MLflowLogger
        from nexus.logger.model_registry import ModelRegistry

        client = MlflowClient(tracking_uri=tracking_uri)
        run_name = f"post_register_{int(time.time())}"
        model_name = f"nexus_smoke_post_register_{int(time.time())}"

        # ── 10a. Create source run + upload a fake best.pth ───────────────────
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt_src = tmp_path / "scratch.pth"
            ckpt_src.write_bytes(b"fake-checkpoint-bytes")

            logger = MLflowLogger(
                run_name=run_name,
                tracking_uri=tracking_uri,
                experiment_name="nexus_smoke_test",
                tags={"researcher": "smoke_test"},
            )
            logger.log_checkpoint(str(ckpt_src), kind="best")
            logger.close()
        ok(f"Source run created: {run_name}")

        # ── 10b. resolve_checkpoint_source — pre-flight without registering ──
        registry = ModelRegistry(tracking_uri=tracking_uri)
        resolved = registry.resolve_checkpoint_source(
            experiment="nexus_smoke_test", run_name=run_name, kind="best"
        )
        if not resolved["source"].endswith("/checkpoints/best.pth"):
            fail(f"Unexpected source URI: {resolved['source']!r}")
            return False
        ok(f"resolve_checkpoint_source returned {resolved['source']}")

        # Missing-artifact path raises FileNotFoundError
        try:
            registry.resolve_checkpoint_source(
                experiment="nexus_smoke_test", run_name=run_name, kind="last"
            )
            fail("resolve_checkpoint_source should have raised for missing 'last'")
            return False
        except FileNotFoundError:
            ok("resolve_checkpoint_source raises on missing artifact")

        # ── 10c. register_from_run_name — actually register + Staging ────────
        result = registry.register_from_run_name(
            experiment="nexus_smoke_test",
            run_name=run_name,
            model_name=model_name,
            kind="best",
            description="smoke test post-hoc registration",
            stage="Staging",
        )
        if not result["version"]:
            fail(f"register_from_run_name returned empty version: {result!r}")
            return False
        ok(f"register_from_run_name returned version {result['version']}")

        # ── 10d. Verify on the server ────────────────────────────────────────
        mv = client.get_model_version(model_name, result["version"])
        if mv.current_stage != "Staging":
            fail(f"Expected stage 'Staging', got {mv.current_stage!r}")
            return False
        ok("New version is in Staging")

        if mv.tags.get("nexus.sourceRunName") != run_name:
            fail(f"nexus.sourceRunName tag missing or wrong: {mv.tags!r}")
            return False
        ok("nexus.sourceRunName tag stamped")

        if mv.description != "smoke test post-hoc registration":
            fail(f"description mismatch: {mv.description!r}")
            return False
        ok("description applied")

        return True
    except Exception as e:
        fail(f"Post-hoc registration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXUS smoke test")
    parser.add_argument(
        "--tracking-uri",
        default="http://127.0.0.1:5100",
        help="MLflow tracking URI (default: http://127.0.0.1:5100)",
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Also run advanced feature tests (SweepLogger, RL metrics)",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}{'=' * 55}")
    print(f"  NEXUS Smoke Test{'  [+advanced]' if args.advanced else ''}")
    print(f"  MLflow URI: {args.tracking_uri}")
    print(f"{'=' * 55}{RESET}")

    results = {}

    results["imports"] = check_imports()
    results["mlflow_connect"] = check_mlflow_connection(args.tracking_uri)

    if not results["mlflow_connect"]:
        print(f"\n{RED}{BOLD}Cannot connect to MLflow server — skipping remaining tests.{RESET}")
        print(f"  → Run bash scheduled_sync/start_local_mlflow.sh first.\n")
        sys.exit(1)

    results["mlflow_logger"] = test_mlflow_logger(args.tracking_uri)
    results["split_params"] = test_split_params(args.tracking_uri)
    results["make_logger"] = test_make_logger_factory(args.tracking_uri)
    results["dual_logger"] = test_dual_logger(args.tracking_uri)

    if args.advanced:
        results["omegaconf_flatten"] = test_omegaconf_flatten(args.tracking_uri)
        results["scheduled_sync"] = test_scheduled_sync_roundtrip(args.tracking_uri)
        results["sweep_logger"] = test_sweep_logger(args.tracking_uri)
        results["eval_logger"] = test_eval_logger(args.tracking_uri)
        results["post_register"] = test_post_register(args.tracking_uri)

    # ── Summary ───────────────────────────────────────────────────────────────
    section("Summary")
    labels = {
        "imports": "Package imports",
        "mlflow_connect": "MLflow server connection",
        "mlflow_logger": "MLflowLogger logging",
        "split_params": "agent_params / env_params split (prefixed params + artifacts)",
        "make_logger": "make_logger factory",
        "dual_logger": "DualLogger (Dual)",
        "omegaconf_flatten": "OmegaConf DictConfig flatten",
        "scheduled_sync": "scheduled_sync round-trip (export -> import)",
        "sweep_logger": "SweepLogger (parent-child runs)",
        "eval_logger": "EvalLogger (eval artifact upload)",
        "post_register": "Model Registry post-hoc registration (register_from_run_name)",
    }
    all_passed = True
    for key, label in labels.items():
        if key not in results:
            continue
        status = PASS if results[key] else FAIL
        print(f"  {status}  {label}")
        if not results[key]:
            all_passed = False

    print()
    if all_passed:
        print(f"  {GREEN}{BOLD}All tests passed! NEXUS is working correctly.{RESET}")
        print(f"  MLflow UI: {args.tracking_uri.replace('127.0.0.1', 'localhost')}")
        print(f"  Experiment: nexus_smoke_test\n")
        sys.exit(0)
    else:
        print(f"  {RED}{BOLD}Some tests failed. Check the error messages above.{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
