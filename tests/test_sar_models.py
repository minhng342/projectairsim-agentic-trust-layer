"""Phase 1 mission / scenario contracts (atal/core/sar.py): the deck's YAML loads, derived
values match slide 13, and bad files are rejected with the field named."""
import pytest
import yaml

from atal.core.action import ActionType, ProposedAction
from atal.core.sar import (
    PlanFileError,
    ReturnHome,
    SweepSector,
    load_mission,
    load_scenario,
    parse_sector_id,
    sector_id,
)

DECK = "missions/sar/mission.yaml"


def deck_dict():
    with open(DECK, encoding="utf-8") as f:
        return yaml.safe_load(f)


def write(tmp_path, data, name="m.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_deck_mission_loads_with_slide_13_derived_values():
    m = load_mission(DECK)
    assert m.sector_size_m == (100.0, 100.0)
    assert m.lane_spacing_m == pytest.approx(48.0)
    assert m.passes_per_sector == 3
    assert len(m.grid.ids()) == 64 and m.grid.last_id == "H8"


def test_deck_scenario_loads_and_keeps_the_victim_out_of_the_mission():
    m = load_mission(DECK)
    sc = load_scenario("missions/sar/scenario.yaml", m)
    assert (sc.victim.x_m, sc.victim.y_m) == (430, 560) and sc.runs == 10
    assert m.grid.sector_at(430, 560) == "F5"
    assert "victim" not in m.model_dump()


def test_dev_files_load():
    m = load_mission("missions/sar/mission_dev.yaml")
    sc = load_scenario("missions/sar/scenario_dev.yaml", m)
    assert m.passes_per_sector == 3 and sc.randomize_victim
    victims = {(v.x_m, v.y_m) for v in (sc.victim_for_run(m, i) for i in range(10))}
    assert len(victims) == 10                                   # a different spot per run
    assert sc.victim_for_run(m, 3) == sc.victim_for_run(m, 3)   # but repeatable


@pytest.mark.parametrize("path, value, field", [
    (("area", "size_m", "width"), 805, "area"),                 # not divisible by 8
    (("sensor", "footprint_m"), 120, "footprint_m"),            # larger than a sector
    (("search", "start_sector"), "J1", "start_sector"),         # outside the grid
    (("search", "start_sector"), "a1", "search.start_sector"),  # malformed
    (("flight", "speed_m_s"), 0, "flight.speed_m_s"),
    (("sensor", "overlap"), 1.0, "sensor.overlap"),
])
def test_bad_missions_are_rejected_naming_the_field(tmp_path, path, value, field):
    d = deck_dict()
    node = d
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    with pytest.raises(PlanFileError) as err:
        load_mission(write(tmp_path, d))
    assert field in str(err.value)


def test_unknown_fields_are_rejected(tmp_path):
    d = deck_dict()
    d["flight"]["turbo"] = True
    with pytest.raises(PlanFileError, match="flight.turbo"):
        load_mission(write(tmp_path, d))


def test_scenario_needs_a_victim_and_checks_it_against_the_mission(tmp_path):
    m = load_mission(DECK)
    base = {"scenario_id": "s", "drones": [{"id": "A", "start": {"x_m": 0, "y_m": 0}}]}
    with pytest.raises(PlanFileError, match="victim"):
        load_scenario(write(tmp_path, base, "s.yaml"), m)
    with pytest.raises(PlanFileError, match="outside the mission area"):
        load_scenario(write(tmp_path, base | {"victim": {"x_m": 900, "y_m": 0}}, "s.yaml"), m)


def test_sector_ids_read_like_a_chess_board_on_a_north_up_map():
    m = load_mission(DECK)
    g = m.grid
    b1, a2 = g.sector("B1"), g.sector("A2")
    assert (b1.y_min, b1.x_min) == (100, 0)          # B1 is east of A1
    assert (a2.x_min, a2.y_min) == (100, 0)          # A2 is north of A1
    assert parse_sector_id(sector_id(4, 5)) == (4, 5)
    assert g.sector_at(-1, 0) is None and g.sector_at(800, 800) == "H8"
    assert g.serpentine()[:9] == ["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1", "H2"]
    assert g.serpentine("C3")[0] == "C3" and sorted(g.serpentine("C3")) == sorted(g.ids())
    with pytest.raises(KeyError):
        g.sector("I1")


def test_typed_actions_become_port_actions():
    p = SweepSector(sector_id="F5", reason="next").to_proposed("Drone1")
    assert p.action_type == ActionType.SWEEP_SECTOR and p.sector_id == "F5"
    assert ReturnHome(reason="done").to_proposed("Drone1").action_type == ActionType.RETURN_HOME
    with pytest.raises(ValueError):
        ProposedAction(vehicle_id="Drone1", action_type=ActionType.SWEEP_SECTOR, reason="x")
    with pytest.raises(ValueError):
        ProposedAction(vehicle_id="Drone1", action_type=ActionType.HOLD, sector_id="A1",
                       reason="sector not allowed for hold")
    with pytest.raises(ValueError):
        SweepSector(sector_id="A1; land", reason="injection")
