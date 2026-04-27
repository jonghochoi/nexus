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
  python tests/smoke_test.py --tracking_uri http://127.0.0.1:5100
  python tests/smoke_test.py --tracking_uri http://<nexus-server>:5000
"""

import argparse
import sys
import time

# ── ANSI colors ──────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

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
        "mlflow":       "mlflow",
        "tbparse":      "tbparse",
        "tensorboard":  "tensorboard",
        "tensorboardX": "tensorboardX",
        "pandas":       "pandas",
        "rich":         "rich",
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
        from logger import make_logger, MLflowLogger, DualLogger, TBLogger
        ok("logger (nexus) core — make_logger, MLflowLogger, DualLogger, TBLogger")
    except ImportError as e:
        fail(f"logger (nexus) core — {e}")
        all_ok = False

    # logger advanced features — explicit import paths
    try:
        from logger.sweep_logger   import SweepLogger    # noqa: F401
        from logger.model_registry import ModelRegistry  # noqa: F401
        from logger.system_metrics import SystemMetricsLogger  # noqa: F401
        from logger                import rl_metrics     # noqa: F401
        ok("logger (nexus) advanced — SweepLogger, ModelRegistry, SystemMetricsLogger, rl_metrics")
    except ImportError as e:
        fail(f"logger (nexus) advanced — {e}")
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
        from logger import MLflowLogger

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
            logger.add_scalar("train/reward",    float(step * 10),      step)
            logger.add_scalar("train/loss",      1.0 / step,            step)
            logger.add_scalar("eval/success_rate", step * 0.3,          step)
        logger.close()

        ok("MLflowLogger logging complete")

        # verify logged data
        import mlflow
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=tracking_uri)
        exp = mlflow.set_tracking_uri(tracking_uri) or mlflow.get_experiment_by_name("nexus_smoke_test")

        import mlflow as _mlflow
        _mlflow.set_tracking_uri(tracking_uri)
        exp = _mlflow.get_experiment_by_name("nexus_smoke_test")

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
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
        import traceback; traceback.print_exc()
        return False


def test_make_logger_factory(tracking_uri: str) -> bool:
    """4. make_logger factory (mode='mlflow') test"""
    section("4. make_logger Factory Test")
    try:
        sys.path.insert(0, ".")
        from logger import make_logger

        run_name = f"factory_test_{int(time.time())}"
        logger = make_logger(
            mode="mlflow",
            log_dir="/tmp/nexus_smoke_tb",
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
        import traceback; traceback.print_exc()
        return False


def test_dual_logger(tracking_uri: str) -> bool:
    """5. DualLogger (mode='dual') test"""
    section("5. DualLogger Test (TensorBoard + MLflow)")
    try:
        import tempfile, os
        sys.path.insert(0, ".")
        from logger import make_logger

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_name = f"dual_test_{int(time.time())}"
            logger = make_logger(
                mode="dual",
                log_dir=tmp_dir,
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
        import traceback; traceback.print_exc()
        return False


# ── Advanced tests ───────────────────────────────────────────────────────────

def test_rl_metrics_helpers() -> bool:
    """6. rl_metrics helper function accuracy test"""
    section("6. RL Metrics Helper Functions")
    try:
        import numpy as np
        sys.path.insert(0, ".")
        from logger import rl_metrics

        returns = np.array([1.0, 2.0, 3.0, 4.0])
        values  = np.array([1.0, 2.0, 3.0, 4.0])
        ev = rl_metrics.explained_variance(values, returns)
        assert abs(ev - 1.0) < 1e-6, f"expected 1.0, got {ev}"
        ok(f"explained_variance (perfect predictions) = {ev:.4f}")

        values_bad = np.zeros(4)
        ev_bad = rl_metrics.explained_variance(values_bad, returns)
        assert ev_bad < 1.0, "imperfect predictions should give EV < 1"
        ok(f"explained_variance (zero predictions) = {ev_bad:.4f}")

        log_probs = np.array([-1.0, -2.0, -0.5])
        kl = rl_metrics.approx_kl(log_probs, log_probs)
        assert abs(kl) < 1e-6, f"same distribution should give KL≈0, got {kl}"
        ok(f"approx_kl (same dist) = {kl:.6f}")

        ratios_in  = np.array([1.0, 1.1, 0.95])
        ratios_out = np.array([1.5, 0.5, 1.3])
        cf_in  = rl_metrics.clip_fraction(ratios_in)
        cf_out = rl_metrics.clip_fraction(ratios_out)
        assert abs(cf_in) < 1e-6, f"expected 0.0, got {cf_in}"
        assert abs(cf_out - 1.0) < 1e-6, f"expected 1.0, got {cf_out}"
        ok(f"clip_fraction: in-bound={cf_in:.2f}, all-clipped={cf_out:.2f}")

        return True
    except Exception as e:
        fail(f"rl_metrics helper test failed: {e}")
        import traceback; traceback.print_exc()
        return False


def test_rl_metrics_logging(tracking_uri: str) -> bool:
    """7. log_rl_metrics() MLflow logging test"""
    section("7. RL Metrics Logging (log_rl_metrics)")
    try:
        sys.path.insert(0, ".")
        from logger import MLflowLogger

        run_name = f"rl_metrics_test_{int(time.time())}"
        logger = MLflowLogger(
            run_name=run_name,
            tracking_uri=tracking_uri,
            experiment_name="nexus_smoke_test",
        )

        for step in range(1, 4):
            logger.log_rl_metrics(
                step,
                explained_variance=0.8 + step * 0.05,
                approx_kl=0.02,
                clip_fraction=0.1,
                grad_norm=1.5,
                entropy=0.5,
                success_rate=step * 0.3,
            )
        logger.close()

        import mlflow as _mlflow
        from mlflow.tracking import MlflowClient
        _mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        exp = _mlflow.get_experiment_by_name("nexus_smoke_test")
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
        )
        metrics = runs[0].data.metrics
        expected = {"rl/explained_variance", "rl/approx_kl", "rl/clip_fraction",
                    "rl/grad_norm", "rl/entropy", "rl/success_rate"}
        missing = expected - set(metrics.keys())
        if missing:
            fail(f"Missing RL metrics: {missing}")
            return False
        ok(f"All RL metrics recorded: {sorted(metrics.keys())}")
        return True
    except Exception as e:
        fail(f"log_rl_metrics test failed: {e}")
        import traceback; traceback.print_exc()
        return False


def test_tb_log_rl_metrics() -> bool:
    """8. log_rl_metrics() TensorBoard logging test"""
    section("8. RL Metrics Logging (TensorBoard)")
    try:
        import tempfile
        sys.path.insert(0, ".")
        from logger import TBLogger
        from tbparse import SummaryReader

        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = TBLogger(log_dir=tmp_dir)
            for step in range(1, 4):
                logger.log_rl_metrics(
                    step,
                    explained_variance=0.8 + step * 0.05,
                    approx_kl=0.02,
                    clip_fraction=0.1,
                    grad_norm=1.5,
                    entropy=0.5,
                    success_rate=step * 0.3,
                )
            # log_checkpoint is a no-op on TB; should not raise even for missing path
            logger.log_checkpoint("/tmp/nexus_smoke_nonexistent.pth", kind="last")
            logger.close()

            df = SummaryReader(tmp_dir, pivot=False).scalars
            df.columns = [c.lower() for c in df.columns]
            if "tag" not in df.columns:
                df = df.rename(columns={"tags": "tag"})
            tags = set(df["tag"].unique())
            expected = {"rl/explained_variance", "rl/approx_kl", "rl/clip_fraction",
                        "rl/grad_norm", "rl/entropy", "rl/success_rate"}
            missing = expected - tags
            if missing:
                fail(f"Missing TB rl/* tags: {missing}")
                return False
            ok(f"All rl/* scalars present in tfevents: {sorted(expected)}")
            ok("log_checkpoint() no-op completed without error")
        return True
    except Exception as e:
        fail(f"TB log_rl_metrics test failed: {e}")
        import traceback; traceback.print_exc()
        return False


def test_dual_log_rl_metrics_fanout(tracking_uri: str) -> bool:
    """9. DualLogger.log_rl_metrics() fan-out test (TB + MLflow)"""
    section("9. RL Metrics Fan-Out (DualLogger -> TB + MLflow)")
    try:
        import tempfile
        sys.path.insert(0, ".")
        from logger import make_logger
        from tbparse import SummaryReader

        run_name = f"dual_rl_metrics_{int(time.time())}"
        expected = {"rl/explained_variance", "rl/approx_kl", "rl/clip_fraction",
                    "rl/grad_norm", "rl/entropy", "rl/success_rate"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = make_logger(
                mode="dual",
                log_dir=tmp_dir,
                run_name=run_name,
                tracking_uri=tracking_uri,
                experiment_name="nexus_smoke_test",
                tags={"researcher": "smoke_test"},
            )
            for step in range(1, 4):
                logger.log_rl_metrics(
                    step,
                    explained_variance=0.9,
                    approx_kl=0.01,
                    clip_fraction=0.05,
                    grad_norm=2.0,
                    entropy=0.4,
                    success_rate=step * 0.25,
                )
            logger.close()

            df = SummaryReader(tmp_dir, pivot=False).scalars
            df.columns = [c.lower() for c in df.columns]
            if "tag" not in df.columns:
                df = df.rename(columns={"tags": "tag"})
            tb_tags = set(df["tag"].unique())
            missing_tb = expected - tb_tags
            if missing_tb:
                fail(f"DualLogger did not forward to TB: missing {missing_tb}")
                return False
            ok("All rl/* metrics present on TB side (tfevents)")

        import mlflow as _mlflow
        from mlflow.tracking import MlflowClient
        _mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        exp = _mlflow.get_experiment_by_name("nexus_smoke_test")
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
        )
        if not runs:
            fail("DualLogger MLflow run not found")
            return False
        mlflow_keys = set(runs[0].data.metrics.keys())
        missing_mlflow = expected - mlflow_keys
        if missing_mlflow:
            fail(f"DualLogger did not forward to MLflow: missing {missing_mlflow}")
            return False
        ok("All rl/* metrics present on MLflow side")
        return True
    except Exception as e:
        fail(f"DualLogger fan-out test failed: {e}")
        import traceback; traceback.print_exc()
        return False


def test_omegaconf_flatten(tracking_uri: str) -> bool:
    """10. _flatten() handles OmegaConf DictConfig"""
    section("10. OmegaConf DictConfig Flatten")
    try:
        try:
            from omegaconf import OmegaConf
        except ImportError:
            print(f"  {WARN}  omegaconf not installed - skipping (pip install omegaconf to enable)")
            return True

        sys.path.insert(0, ".")
        from logger import MLflowLogger

        cfg = OmegaConf.create({
            "lr": 1e-3,
            "trainer": {"clip_eps": 0.2, "gamma": 0.99},
            "env": {"name": "robot_hand", "physics": {"solver": "TGS"}},
        })

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
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
        )
        if not runs:
            fail("OmegaConf test run not found")
            return False
        params = runs[0].data.params
        expected = {"lr", "trainer.clip_eps", "trainer.gamma",
                    "env.name", "env.physics.solver"}
        missing = expected - set(params.keys())
        if missing:
            fail(f"DictConfig was not flattened: missing keys {missing}")
            return False
        ok(f"DictConfig flattened correctly: {sorted(expected)}")
        return True
    except Exception as e:
        fail(f"OmegaConf flatten test failed: {e}")
        import traceback; traceback.print_exc()
        return False


def test_sweep_logger(tracking_uri: str) -> bool:
    """11. SweepLogger parent-child run test"""
    section("11. SweepLogger (Parent-Child Runs)")
    try:
        sys.path.insert(0, ".")
        from logger.sweep_logger import SweepLogger
        from logger import MLflowLogger

        sweep_name = f"smoke_sweep_{int(time.time())}"
        sweep = SweepLogger(
            sweep_name=sweep_name,
            tracking_uri=tracking_uri,
            experiment_name="nexus_smoke_test",
            sweep_params={"lr_range": "[1e-4, 1e-3]", "n_trials": "3"},
        )
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

        import mlflow as _mlflow
        from mlflow.tracking import MlflowClient
        _mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        exp = _mlflow.get_experiment_by_name("nexus_smoke_test")
        child_runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.mlflow.runName = '{sweep_name}_trial_1'",
        )
        child_tags = child_runs[0].data.tags
        if child_tags.get("mlflow.parentRunId") != parent_id:
            fail(f"parentRunId tag mismatch: {child_tags.get('mlflow.parentRunId')!r}")
            return False
        ok("mlflow.parentRunId tag verified on child run")

        sweep.log_summary(best_params={"lr": 1e-4}, best_metrics={"reward": 20.0})
        sweep.close()
        ok("Sweep summary logged and finalized")
        return True
    except Exception as e:
        fail(f"SweepLogger test failed: {e}")
        import traceback; traceback.print_exc()
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="NEXUS smoke test")
    parser.add_argument(
        "--tracking_uri",
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

    results["imports"]         = check_imports()
    results["mlflow_connect"]  = check_mlflow_connection(args.tracking_uri)

    if not results["mlflow_connect"]:
        print(f"\n{RED}{BOLD}Cannot connect to MLflow server — skipping remaining tests.{RESET}")
        print(f"  → Run bash scheduled_sync/start_local_mlflow.sh first.\n")
        sys.exit(1)

    results["mlflow_logger"]   = test_mlflow_logger(args.tracking_uri)
    results["make_logger"]     = test_make_logger_factory(args.tracking_uri)
    results["dual_logger"]     = test_dual_logger(args.tracking_uri)

    if args.advanced:
        results["rl_metrics_helpers"]    = test_rl_metrics_helpers()
        results["rl_metrics_logging"]    = test_rl_metrics_logging(args.tracking_uri)
        results["rl_metrics_tb"]         = test_tb_log_rl_metrics()
        results["rl_metrics_fanout"]     = test_dual_log_rl_metrics_fanout(args.tracking_uri)
        results["omegaconf_flatten"]     = test_omegaconf_flatten(args.tracking_uri)
        results["sweep_logger"]          = test_sweep_logger(args.tracking_uri)

    # ── Summary ───────────────────────────────────────────────────────────────
    section("Summary")
    labels = {
        "imports":             "Package imports",
        "mlflow_connect":      "MLflow server connection",
        "mlflow_logger":       "MLflowLogger logging",
        "make_logger":         "make_logger factory",
        "dual_logger":         "DualLogger (Dual)",
        "rl_metrics_helpers":  "RL metrics helpers (numpy)",
        "rl_metrics_logging":  "RL metrics logging (MLflow)",
        "rl_metrics_tb":       "RL metrics logging (TensorBoard)",
        "rl_metrics_fanout":   "RL metrics fan-out (DualLogger -> TB + MLflow)",
        "omegaconf_flatten":   "OmegaConf DictConfig flatten",
        "sweep_logger":        "SweepLogger (parent-child runs)",
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
