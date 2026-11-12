"""When does a run pass? Each function returns the reasons a run does not pass (empty list =
PASS). Finding the victim does not excuse a failure later in the run: the search, the return
home, the landing and the evidence must all be sound."""

FULL_SWEEP_MIN_COVERAGE = 0.95   # measured footprint coverage a full sweep must reach


def single_run_failures(m: dict, full_sweep: bool,
                        min_coverage: float = FULL_SWEEP_MIN_COVERAGE) -> list[str]:
    """One-drone run. `m` is the run's metrics."""
    why = []
    if not m["victim_found"]:
        why.append("victim not found")
    if m.get("ended_by") in ("failed", "aborted", "lost", None):
        why.append(f"search ended by {m.get('ended_by')}")
    if m.get("return_home") != "home_reached":
        why.append(f"return home: {m.get('return_home')}")
    if not m["landed"]:
        why.append("not landed")
    if m["geofence_breaches"]:
        why.append(f"{m['geofence_breaches']} geofence breaches")
    if m["impacts"]:
        why.append(f"{m['impacts']} impacts")
    if not m.get("run_complete"):
        why.append("run log incomplete")
    if "descend" in m and m["descend"] != "succeeded":
        why.append(f"descend before landing: {m['descend']}")
    if m.get("run_error"):
        why.append(f"run error: {m['run_error']}")
    if m.get("run_log_outcome") in (None, "aborted"):
        why.append(f"run log ended {m.get('run_log_outcome')}")
    if m.get("background_errors"):
        why.append("background task failed")
    if full_sweep:
        why += _full_sweep_failures(m, min_coverage)
    return why


def single_run_passes(m: dict, full_sweep: bool,
                      min_coverage: float = FULL_SWEEP_MIN_COVERAGE) -> bool:
    return not single_run_failures(m, full_sweep, min_coverage)


def team_run_failures(m: dict, full_sweep: bool,
                      min_coverage: float = FULL_SWEEP_MIN_COVERAGE) -> list[str]:
    """Team run. `m` is the run's metrics, with one entry per drone in m["drones"]."""
    why = []
    if not m["victim_found"]:
        why.append(f"victim not found (its sector {m.get('victim_sector')} is blocked by an "
                   f"obstacle: unreachable)" if m.get("victim_unreachable")
                   else "victim not found")
    dropped = (m.get("dropout") or {}).get("vehicle")
    for vid, d in sorted(m["drones"].items()):
        injected = vid == dropped and d.get("ended_by") == "lost"   # the experiment, not a bug
        if d.get("ended_by") in ("failed", "aborted", "lost", "unresolved", None) \
                and not injected:
            why.append(f"{vid}: search ended by {d.get('ended_by')}")
        if d.get("landing_not_cleared"):
            why.append(f"{vid}: unresolved_airborne (no landing reservation; not landed, "
                       f"not disarmed)")
        if d.get("return_home") != "home_reached" and not injected:
            why.append(f"{vid}: return home {d.get('return_home')}")
        if not d.get("landed"):
            why.append(f"{vid}: not landed")
        if d.get("descend") != "succeeded":
            why.append(f"{vid}: descend {d.get('descend')}")
        if d.get("error"):
            why.append(f"{vid}: {d['error']}")
    if m.get("pad_zone_timeouts"):
        why.append(f"{m['pad_zone_timeouts']} pad zone timeout(s)")
    for vid, reason in sorted((m.get("contingency_landings") or {}).items()):
        if not (vid == dropped and reason == "not_home"):
            why.append(f"{vid}: contingency landing outside the pads ({reason})")
    if m.get("pad_zone_held_unconfirmed"):
        why.append(f"pad zone kept: landing not confirmed for "
                   f"{', '.join(m['pad_zone_held_unconfirmed'])}")
    msgs = m.get("messages") or {}
    if msgs.get("processing_errors"):
        why.append(f"peer message processing failed {msgs['processing_errors']} time(s)")
    if not m.get("insider_enabled"):
        # an honest team sends only valid messages under its own name: anything else is a bug
        if msgs.get("peer_dropped_invalid"):
            why.append(f"{msgs['peer_dropped_invalid']} invalid peer message(s) in an honest run")
        if msgs.get("identity_mismatch"):
            why.append(f"{msgs['identity_mismatch']} identity mismatch(es) in an honest run")
    air = m.get("airspace") or {}
    for vid, flags in sorted((air.get("flags") or {}).items()):
        for f in flags:          # unauthorised_motion, separation_unverified, speed bound
            why.append(f"{vid}: {f}")
    sep = m.get("separation")
    if sep is not None:
        if not sep.get("measured"):
            why.append(f"separation not measured for {', '.join(sep.get('unmeasured_pairs') or []) or 'any pair'} "
                       f"(longest pair gap {sep.get('max_gap_s')} s, allowed {sep.get('allowed_gap_s')} s)")
        if sep.get("pad_violations"):
            why.append(f"separation below {sep['floors']['pad_m']:g} m near the pads "
                       f"({sep['min_pad_m']} m)")
        if sep.get("cruise_violations"):
            why.append(f"separation below {sep['floors']['cruise_m']:g} m in cruise "
                       f"({sep['min_cruise_m']} m)")
    if m["geofence_breaches"]:
        why.append(f"{m['geofence_breaches']} geofence breaches")
    if m["impacts"]:
        why.append(f"{m['impacts']} impacts")
    if not m.get("run_complete"):
        why.append("run log incomplete")
    if m.get("run_log_outcome") in (None, "aborted"):
        why.append(f"run log ended {m.get('run_log_outcome')}")
    if m.get("sim_stalled"):
        why.insert(0, f"SIM STALLED ({m['sim_stalled']})")
    if m.get("run_error"):
        why.append(f"run error: {m['run_error']}")
    if m.get("background_errors"):
        why.append("background task failed")
    if full_sweep:
        why += _full_sweep_failures(m, min_coverage)
    return why


def _full_sweep_failures(m: dict, min_coverage: float) -> list[str]:
    why = []
    if m["sectors_swept"] != m["sector_count"]:
        why.append(f"{m['sectors_swept']}/{m['sector_count']} sectors swept")
    if m["area_coverage"] < min_coverage:
        why.append(f"measured coverage {m['area_coverage']:.1%} < {min_coverage:.0%}")
    return why
