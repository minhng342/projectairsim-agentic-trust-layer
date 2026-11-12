"""
Top-down track plot of a run, as a standalone SVG (no plotting library needed).

    python -m atal.evaluation.plot_tracks runs/run_<id>.jsonl [-o out.svg]

Works on any run log with `truth` events (ground-truth poses, evaluation side):
    corridor runs (Phase 0, legacy/corridor)        planned corridors (dashed) + flown tracks
    SAR runs (run_sar_mission.py)                    sector grid with ids, sectors shaded by
                                                     the drone's memory (swept / partial),
                                                     flown track, true victim, detection point
North is up, east is right (x = north, y = east in the NED frame). Open the SVG in a browser.
"""
import argparse
import html
import math
import sys
from pathlib import Path

from atal.evaluation.run_log import read_run

COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be185d",
          "#4d7c0f", "#a16207", "#475569"]


def _tracks(log):
    out = {}
    for e in log.of_kind("truth"):
        for vid, p in (e.get("poses") or {}).items():
            out.setdefault(vid, []).append((p["north_m"], p["east_m"], p.get("altitude_m")))
    return out


class _Canvas:
    def __init__(self, x_min, x_max, y_min, y_max, size_px=900, pad_px=50):
        span = max(x_max - x_min, y_max - y_min, 1e-6)
        self.s = (size_px - 2 * pad_px) / span
        self.pad = pad_px
        self.x_max, self.y_min = x_max, y_min
        self.w = (y_max - y_min) * self.s + 2 * pad_px
        self.h = (x_max - x_min) * self.s + 2 * pad_px + 40
        self.items: list[str] = []

    def p(self, x, y):
        """world (x north, y east) -> svg (px right, py down)."""
        return self.pad + (y - self.y_min) * self.s, self.pad + (self.x_max - x) * self.s

    def add(self, s):
        self.items.append(s)

    def svg(self, title):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w:.0f}" height="{self.h:.0f}" '
                f'viewBox="0 0 {self.w:.0f} {self.h:.0f}" font-family="sans-serif">'
                f'<rect width="100%" height="100%" fill="#ffffff"/>'
                f'<text x="{self.pad}" y="28" font-size="16" fill="#111">{html.escape(title)}</text>'
                + "".join(self.items) + "</svg>")


def _polyline(c, pts, color, width=1.5, dash=None, opacity=1.0):
    if len(pts) < 2:
        return
    d = " ".join(f"{px:.1f},{py:.1f}" for px, py in (c.p(x, y) for x, y in pts))
    c.add(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
          f'stroke-opacity="{opacity}" stroke-linejoin="round"'
          + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")


def _decimate(pts, min_step):
    out = []
    for p in pts:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) >= min_step:
            out.append(p)
    if pts and out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


def plot_run(path, out=None) -> Path:
    log = read_run(path)
    cfg = (log.events[0].get("config") or {}) if log.events else {}
    tracks = _tracks(log)
    mission = cfg.get("mission") or {}
    is_sar = "area" in mission
    xs = [p[0] for t in tracks.values() for p in t]
    ys = [p[1] for t in tracks.values() for p in t]
    if is_sar:
        a = mission["area"]
        ox, oy = a["origin"]["x_m"], a["origin"]["y_m"]
        w, h = a["size_m"]["width"], a["size_m"]["height"]
        xs += [ox, ox + w]
        ys += [oy, oy + h]
    else:
        for d in mission.get("drones", []):
            for x, y in d["corridor"]["waypoints"]:
                xs.append(x)
                ys.append(y)
    if not xs:
        raise ValueError(f"{path}: no truth samples or mission geometry to plot")
    m = 0.05 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    c = _Canvas(min(xs) - m, max(xs) + m, min(ys) - m, max(ys) + m)
    step = 0.5 / c.s if c.s else 0.5

    if is_sar:
        from atal.core.mission import Mission
        ms = Mission.model_validate(mission)
        status, swept_by = {}, {}
        team = len(tracks) > 1
        for e in log.of_kind("belief_update"):
            s = e.get("sector")
            if not s:
                continue
            if team:
                # swarm: shade by who swept it ITSELF (peer clears are not sweeps)
                if s.get("verified_by_self") and s["status"] == "swept":
                    swept_by.setdefault(s["sector_id"], []).append(e.get("vehicle_id"))
                elif s.get("verified_by_self") and s["sector_id"] not in swept_by:
                    status[s["sector_id"]] = s["status"]
            else:
                status[s["sector_id"]] = s["status"]
        fill = {"swept": "#bbf7d0", "partial": "#fde68a", "in_progress": "#fde68a"}
        color_of = {vid: COLORS[i % len(COLORS)] for i, vid in enumerate(sorted(tracks))}
        for sec in ms.grid.sectors():
            x0, y0 = c.p(sec.x_max, sec.y_min)
            x1, y1 = c.p(sec.x_min, sec.y_max)
            who = swept_by.get(sec.id)
            if who:
                col, op = color_of.get(who[0], "#94a3b8"), 0.22
            else:
                col, op = fill.get(status.get(sec.id), "#f8fafc"), 1.0
            c.add(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" '
                  f'fill="{col}" fill-opacity="{op}" stroke="#94a3b8" stroke-width="0.8"/>')
            label = sec.id + (" x" + str(len(who)) if who and len(who) > 1 else "")
            c.add(f'<text x="{x0 + 3:.1f}" y="{y0 + 12:.1f}" font-size="10" fill="#64748b">{label}</text>')
        victim = cfg.get("victim")
        det = next(iter(log.of_kind("victim_detected")), None)
    else:
        for i, d in enumerate(mission.get("drones", [])):
            _polyline(c, d["corridor"]["waypoints"], COLORS[i % len(COLORS)], 1.0, "6 4", 0.6)
        victim, det = None, None

    for i, (vid, t) in enumerate(sorted(tracks.items())):
        col = COLORS[i % len(COLORS)]
        _polyline(c, _decimate([(p[0], p[1]) for p in t], step), col, 1.6)
        sx, sy = c.p(t[0][0], t[0][1])
        c.add(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="4" fill="{col}"/>'
              f'<text x="{sx + 6:.1f}" y="{sy - 6:.1f}" font-size="11" fill="{col}">{html.escape(vid)}</text>')
    if victim:
        vx, vy = c.p(victim["x_m"], victim["y_m"])
        c.add(f'<g stroke="#b91c1c" stroke-width="2.5"><line x1="{vx - 7:.1f}" y1="{vy - 7:.1f}" '
              f'x2="{vx + 7:.1f}" y2="{vy + 7:.1f}"/><line x1="{vx - 7:.1f}" y1="{vy + 7:.1f}" '
              f'x2="{vx + 7:.1f}" y2="{vy - 7:.1f}"/></g>'
              f'<text x="{vx + 9:.1f}" y="{vy + 4:.1f}" font-size="11" fill="#b91c1c">victim</text>')
    if det:
        d = det["detection"]
        dx, dy = c.p(d["x_m"], d["y_m"])
        half = d["footprint_m"] / 2 * c.s
        c.add(f'<rect x="{dx - half:.1f}" y="{dy - half:.1f}" width="{2 * half:.1f}" '
              f'height="{2 * half:.1f}" fill="none" stroke="#b91c1c" stroke-dasharray="3 2"/>'
              f'<circle cx="{dx:.1f}" cy="{dy:.1f}" r="4" fill="none" stroke="#b91c1c" stroke-width="2"/>')
    end = log.events[-1] if log.complete else {}
    title = (f"{cfg.get('script', 'run')}  {log.events[0].get('run_id', '') if log.events else ''}"
             f"  outcome: {end.get('outcome', 'incomplete')}")
    if is_sar and len(tracks) > 1:
        legend = ("sector tint = the drone that swept it (xN = swept N times), amber = partial, "
                  "X = true victim, dashed box = detection footprint")
    elif is_sar:
        legend = "green = swept (memory), amber = partial, X = true victim, dashed box = detection footprint"
    else:
        legend = "dashed = planned corridor, solid = flown track (ground truth)"
    c.add(f'<text x="{c.pad}" y="{c.h - 14:.0f}" font-size="11" fill="#475569">'
          f'{html.escape(legend)}. North up, east right.</text>')
    out = Path(out) if out else Path(path).with_suffix(".svg")
    out.write_text(c.svg(title), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("run_log")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    print(plot_run(a.run_log, a.out))
    sys.exit(0)
