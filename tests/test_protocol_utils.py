import pytest

from scripts import watch_training
from scripts.collect_parking import choose_policy_name, parse_policy_mix
from scripts.eval_protocol import CSV_FIELDS, parse_planners, summarize
from scripts.sweep_plan import summarize as summarize_sweep


def test_parse_policy_mix_rejects_empty_and_unknown():
    assert parse_policy_mix("random,pd") == ["random", "pd"]
    assert parse_policy_mix("reverse_expert,near_goal_expert") == ["reverse", "near_goal_correction"]
    with pytest.raises(ValueError):
        parse_policy_mix("")
    with pytest.raises(ValueError):
        parse_policy_mix("random,nope")


def test_choose_policy_name_cycles():
    policies = ["a", "b", "c"]
    assert [choose_policy_name(i, policies) for i in range(5)] == ["a", "b", "c", "a", "b"]


def test_eval_summary_contains_stable_metrics():
    rows = [
        {
            "init_dist": 5.0,
            "final_dist": 3.0,
            "final_ang": 10.0,
            "success": 0,
            "strict_success": 0,
            "abs_lateral_offset_m": 0.5,
            "abs_along_offset_m": 1.5,
            "abs_heading_error_deg": 25.0,
            "collided": 1,
        },
        {
            "init_dist": 6.0,
            "final_dist": 1.0,
            "final_ang": 5.0,
            "success": 1,
            "strict_success": 1,
            "abs_lateral_offset_m": 0.1,
            "abs_along_offset_m": 0.25,
            "abs_heading_error_deg": 3.0,
            "collided": 0,
        },
    ]
    out = summarize(rows)
    assert out["episodes"] == 2
    assert out["mean_final_dist"] == 2.0
    assert out["best_final_dist"] == 1.0
    assert out["success_rate"] == 0.5
    assert out["strict_success_rate"] == 0.5
    assert out["mean_abs_lateral_offset_m"] == pytest.approx(0.3)
    assert out["mean_abs_along_offset_m"] == pytest.approx(0.875)
    assert out["mean_abs_heading_error_deg"] == pytest.approx(14.0)
    assert out["collision_rate"] == 0.5
    assert out["mean_dist_delta"] == 3.5


def test_eval_protocol_csv_fields_include_strict_parking_metrics():
    required = {
        "strict_success",
        "lateral_offset_m",
        "along_offset_m",
        "abs_lateral_offset_m",
        "abs_along_offset_m",
        "heading_error_deg",
        "abs_heading_error_deg",
        "speed_mps",
        "collided",
    }

    assert required.issubset(set(CSV_FIELDS))


def test_sweep_summary_contains_strict_parking_metrics():
    stats = summarize_sweep(
        init_dist=[5.0, 6.0],
        final_dist=[3.0, 1.0],
        final_ang=[10.0, 5.0],
        success=[0.0, 1.0],
        strict_success=[0.0, 0.0],
        abs_lateral_offset_m=[0.8, 0.2],
        abs_along_offset_m=[1.2, 0.3],
        abs_heading_error_deg=[30.0, 10.0],
        collided=[1.0, 0.0],
    )

    assert stats["success_rate"] == 0.5
    assert stats["strict_success_rate"] == 0.0
    assert stats["mean_abs_lateral_offset_m"] == pytest.approx(0.5)
    assert stats["mean_abs_along_offset_m"] == pytest.approx(0.75)
    assert stats["mean_abs_heading_error_deg"] == pytest.approx(20.0)
    assert stats["collision_rate"] == 0.5


def test_parse_eval_planners_rejects_unknown():
    assert parse_planners("random,pd,model,model_pd") == ["random", "pd", "model", "model_pd"]
    with pytest.raises(ValueError):
        parse_planners("")
    with pytest.raises(ValueError):
        parse_planners("random,nope")


def test_watch_training_reads_eval_summary(tmp_path):
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "eval_protocol"
    eval_dir.mkdir(parents=True)
    (run_dir / "train_log.csv").write_text("step,epoch,loss,loss_pred,loss_sigreg,lr,elapsed\n1,0,1,1,0,1e-4,1\n")
    (eval_dir / "summary.json").write_text(
        """
        {
          "episodes": 1,
          "planner_order": ["random", "pd"],
          "planners": {
            "random": {"success_rate": 0.0, "mean_final_dist": 10.0},
            "pd": {"success_rate": 1.0, "mean_final_dist": 1.0}
          }
        }
        """,
        encoding="utf-8",
    )

    watch_training.configure_paths(run_dir, tmp_path / "dash")
    summary = watch_training.load_latest_eval_summary()
    assert summary is not None
    assert summary["episodes"] == 1
    assert summary["planner_order"] == ["random", "pd"]
    assert summary["planners"]["pd"]["success_rate"] == 1.0
