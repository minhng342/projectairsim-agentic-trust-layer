"""Voronoi partition + coverage order, boustrophedon lanes, footprint detection."""
import math

import pytest

from atal.core.mission import load_mission
from atal.search.sweep_lanes import sweep_path
from atal.search.voronoi import coverage_order, voronoi_partition
from atal.simulation.victim_sensor import segment_box_hit

M = load_mission("bundled:missions/sar/mission.yaml")
G = M.grid


def test_one_drone_owns_the_whole_grid_and_flies_it_in_serpentine_order():
    cells = voronoi_partition(G, {"Drone1": (0.0, 0.0)})
    assert sorted(cells["Drone1"]) == sorted(G.ids())
    assert coverage_order(G, cells["Drone1"], (0, 0), "A1") == G.serpentine()


def test_partition_gives_each_sector_to_the_nearest_drone_deterministically():
    pos = {"b": (790.0, 790.0), "a": (10.0, 10.0), "c": (10.0, 790.0)}
    cells = voronoi_partition(G, pos)
    assert sorted(s for v in cells.values() for s in v) == sorted(G.ids())
    for d, own in cells.items():
        for s in own:
            cx, cy = G.sector(s).center
            mine = math.hypot(pos[d][0] - cx, pos[d][1] - cy)
            assert all(mine <= math.hypot(p[0] - cx, p[1] - cy) + 1e-9 for p in pos.values())
    assert "A1" in cells["a"] and "H8" in cells["b"] and "H1" in cells["c"]
    assert voronoi_partition(G, dict(reversed(list(pos.items())))) == cells
    # recomputed over what is left (Phase 3: sectors clear, a drone drops out)
    left = voronoi_partition(G, {"a": (10.0, 10.0)}, available=cells["b"])
    assert left == {"a": cells["b"]}


def test_order_over_an_irregular_cell_has_no_long_jumps():
    cells = voronoi_partition(G, {"a": (0.0, 0.0), "b": (800.0, 800.0), "c": (0.0, 800.0)})
    order = coverage_order(G, cells["b"], (800, 800), "A1")
    assert order[0] == "H8" and sorted(order) == sorted(cells["b"])
    hops = [math.dist(G.sector(p).center, G.sector(q).center) for p, q in zip(order, order[1:])]
    assert max(hops) <= 100 * math.sqrt(2) + 1e-6


def test_lanes_cover_the_sector_with_the_mission_overlap():
    p = sweep_path(M, "C2", from_xy=(100.0, 200.0))
    s = G.sector("C2")
    assert p.lanes == 3 and len(p.waypoints) == 6
    gaps = [b - a for a, b in zip(p.lane_xs, p.lane_xs[1:])]
    assert all(g <= M.lane_spacing_m + 1e-9 for g in gaps)
    half = M.sensor.footprint_m / 2
    assert p.lane_xs[0] - half <= s.x_min and p.lane_xs[-1] + half >= s.x_max
    assert p.lane_y[0] - half <= s.y_min and p.lane_y[1] + half >= s.y_max
    assert math.dist(p.start, (100.0, 200.0)) == min(
        math.dist(c, (100.0, 200.0)) for c in [(x, y) for x in (p.lane_xs[0], p.lane_xs[-1])
                                               for y in p.lane_y])
    for (x0, y0), (x1, y1) in zip(p.waypoints, p.waypoints[1:]):   # axis-aligned legs
        assert x0 == x1 or y0 == y1


def test_detection_box_check_catches_a_fast_pass_between_samples():
    assert segment_box_hit((0, -50), (0, 50), (0, 0), 1.0) == pytest.approx((0, 0))
    assert segment_box_hit((5, -50), (5, 50), (0, 0), 1.0) is None
    assert segment_box_hit((0.5, 0.5), (0.5, 0.5), (0, 0), 1.0) == pytest.approx((0.5, 0.5))
