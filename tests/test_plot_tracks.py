"""Top-down track plot (evaluation/plot_tracks.py) for corridor (Phase 0) run logs."""
from evaluation.plot_tracks import plot_run
from evaluation.recorder import EvaluationRecorder
from legacy.corridor.mission import MissionPlan


def test_corridor_run_plots_planned_corridors_and_flown_tracks(tmp_path):
    plan = MissionPlan.load("missions/inspection_5.json")
    rec = EvaluationRecorder(path=tmp_path / "run.jsonl", include_git=False,
                             run_config={"script": "test_corridors_live.py",
                                         "mission": plan.model_dump(mode="json")})
    for i in range(20):
        rec.record("truth", poses={v: {"sim_time_ns": i, "north_m": float(i), "east_m": float(k),
                                       "altitude_m": 10.0}
                                   for k, v in enumerate(plan.vehicle_ids)})
    rec.close("pass")
    svg = plot_run(tmp_path / "run.jsonl").read_text()
    assert svg.startswith("<svg") and svg.count("<polyline") == 2 * len(plan.vehicle_ids)
    assert "outcome: pass" in svg and "Drone5" in svg
