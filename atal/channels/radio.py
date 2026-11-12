"""
Remote ID broadcast channel: how each drone (and its agent) hears the others.

    RemoteIdBroadcaster (one per drone)        RemoteIdBus                RemoteIdReceiver (one per drone)
      own simulator snapshot (honest)    ->      for every OTHER drone:  ->   stamp id / sequence / receipt time
      -> message (dict)                          copy -> link hooks             stream = CLAIMED uas_id
      -> sender hooks (red: lie, add,                                           trust assessor (stream history)
         drop, impersonate; may emit                                            latest report per claimed id
         several messages)                                                      -> traffic view for the agent

Who knows what (the same split as the observation feed):
    receiver / agent / gate   only the delivered payload and the receiver's stamps; the
                              stream is keyed by the CLAIMED uas_id, never the true sender
    recorder (evaluation)     everything: the true sender, the clean (truthful) message,
                              which hooks changed it, and which receiver got which copy

Sender hooks model a malicious drone (it controls what it transmits). Link hooks model
in-transit interference (per delivery, optionally for one receiver only). A hook is
    hook(payload: dict, ctx: RidContext) -> dict | list[dict] | None
receiving a private copy; None drops it, a list transmits several messages (e.g. fake
IDs). A hook that raises loses the message, never crashes the channel.

The honest broadcaster reads simulator ground truth at a fixed rate (default 1 Hz, the
order of real Remote ID). Its drone's agent never sees its own broadcast.
"""
import asyncio
import copy
import math
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime, timezone

from atal.core.observation import TrustAssessment, TrustStatus
from atal.core.remote_id import RemoteIdReport
from atal.trust_layer.assessor import NoTrustAssessor

CHANNEL = "remote_id"


@dataclass(frozen=True)
class RidContext:
    sender_id: str               # TRUE transmitter (sender hooks) / origin (link hooks)
    sequence: int                # sender's transmission count
    receiver_id: str | None = None   # link hooks only


def _hook_name(hook) -> str:
    return getattr(hook, "name", None) or getattr(hook, "__name__", None) or type(hook).__name__


def message_from_snapshot(uas_id: str, snap, self_id_text: str = "",
                          target_altitude_m: float | None = None,
                          mission_state: str | None = None) -> dict:
    """The honest broadcast for one simulator snapshot. target_altitude_m is an intent
    extension (not in ASTM F3411; modeled on ADS-B's selected altitude): the layer the
    drone is going to, so others don't pick it too. mission_state (Phase 3, modeled on
    F3411's operational status) says whether the drone is still searching, so peers share
    out its area once it heads home. Like every field, they are claims."""
    vn, ve = snap.velocity_ned_mps.x, snap.velocity_ned_mps.y
    gs = math.hypot(vn, ve)
    return {
        "uas_id": uas_id,
        "sim_time_ns": snap.sim_time_ns,
        "north_m": round(snap.position_ned_m.x, 3),
        "east_m": round(snap.position_ned_m.y, 3),
        "altitude_m": round(snap.altitude_local_m, 3),
        "ground_speed_mps": round(gs, 3),
        "track_deg": round(math.degrees(math.atan2(ve, vn)) % 360.0, 1) if gs >= 0.3 else None,
        "vertical_speed_mps": round(snap.vertical_speed_mps, 3),
        "self_id_text": self_id_text,
    } | ({} if target_altitude_m is None else {"target_altitude_m": round(float(target_altitude_m), 2)}) \
      | ({} if mission_state is None else {"mission_state": mission_state})


def _apply_hooks(hooks, payloads: list[dict], ctx: RidContext, on_error):
    """Run hooks over each message; returns (messages, names of hooks that changed any)."""
    changed = []
    for hook in hooks:
        out = []
        for p in payloads:
            before = copy.deepcopy(p)
            try:
                r = hook(copy.deepcopy(p), ctx)
            except Exception as err:
                on_error(hook, err)
                continue
            r = [] if r is None else (r if isinstance(r, list) else [r])
            if r != [before] and _hook_name(hook) not in changed:
                changed.append(_hook_name(hook))
            out.extend(x for x in r if x is not None)
        payloads = out
    return payloads, changed


class RemoteIdReceiver:
    def __init__(self, owner_id: str, assessor=None, recorder=None, clock=time.monotonic,
                 remember: int = 2000, history_per_stream: int = 30):
        self.owner_id = owner_id
        self.clock = clock
        self._assessor = assessor or NoTrustAssessor()
        self._recorder = recorder
        self._latest: dict[str, RemoteIdReport] = {}
        self._history: dict[str, deque[RemoteIdReport]] = {}
        self._history_len = history_per_stream
        self._seq: dict[str, int] = {}
        self._issued: OrderedDict[str, RemoteIdReport] = OrderedDict()
        self._remember = remember

    def receive(self, payload) -> RemoteIdReport | None:
        """Called by the bus for each delivered message. Returns the report (for the bus's
        evaluation record), or None if it could not be read at all."""
        if not isinstance(payload, dict):
            self._record("rid_unreadable", reason=f"payload is {type(payload).__name__}")
            return None
        claimed = payload.get("uas_id")
        claimed = claimed if isinstance(claimed, str) and claimed else "<missing>"
        self._seq[claimed] = self._seq.get(claimed, 0) + 1
        hist = self._history.setdefault(claimed, deque(maxlen=self._history_len))
        report = RemoteIdReport(receiver_id=self.owner_id, channel=CHANNEL,
                                claimed_uas_id=claimed, sequence=self._seq[claimed],
                                received_host_s=self.clock(),
                                received_at_utc=datetime.now(timezone.utc),
                                payload=copy.deepcopy(payload))
        previous = hist[-1].model_copy(deep=True) if hist else None
        try:
            assessment = self._assessor.assess(report.model_copy(deep=True), previous)
        except Exception as err:
            assessment = TrustAssessment(
                observation_id=report.observation_id, assessor=getattr(self._assessor, "name", "?"),
                assessed=False, status=TrustStatus.NOT_ASSESSED,
                reasons=[f"assessor raised {type(err).__name__}: {err}"])
        report = report.model_copy(update={"assessment": assessment})
        hist.append(report)
        self._latest[claimed] = report
        self._issued[report.observation_id] = report
        while len(self._issued) > self._remember:
            self._issued.popitem(last=False)
        self._record("rid_report", report=report.model_dump(mode="json"))
        return report

    # ------------------------------------------------------------ agent / gate side
    def latest(self) -> dict[str, RemoteIdReport]:
        """Newest report per CLAIMED id (deep copies), including stale ones."""
        return {k: v.model_copy(deep=True) for k, v in self._latest.items()}

    def history(self, claimed_uas_id: str, n: int | None = None) -> list[RemoteIdReport]:
        items = list(self._history.get(claimed_uas_id, ()))
        if n is not None:
            items = items[-n:] if n > 0 else []
        return [r.model_copy(deep=True) for r in items]

    def resolve(self, report_id) -> RemoteIdReport | None:
        if not isinstance(report_id, str):
            return None
        r = self._issued.get(report_id)
        return None if r is None else r.model_copy(deep=True)

    def _record(self, kind, **fields):
        if self._recorder is not None:
            self._recorder.record(kind, vehicle_id=self.owner_id, **fields)


class RemoteIdBus:
    """In-process broadcast medium. Every message reaches every attached receiver except
    the true sender's own, and (Phase 3) only receivers within radio range.

    in_range(true_sender_id, receiver_id) -> (bool, distance_m | None) decides range on the
    evaluation side, from the simulator's TRUE positions (radio physics does not care what
    a message claims). A message out of range is not delivered and is recorded as
    rid_out_of_range. None = every receiver hears everything (Phase 0-2).

    A receiver is anything with `owner_id` and `receive(payload)`; the swarm receiver
    (atal/agents/swarm.py) takes Remote ID reports and typed peer messages on the same bus.
    A receiver with `receive_linked(payload, link_sender)` is also told the transmitter's
    authenticated identity (Phase 3: the link layer of an authenticated radio). It is the
    transmitter's enrolled id, never its position or whether it is honest."""

    def __init__(self, recorder=None, in_range=None):
        self._receivers: dict[str, RemoteIdReceiver] = {}
        self._link_hooks: list[tuple[str | None, object]] = []
        self._recorder = recorder
        self._in_range = in_range
        self._tx = 0
        self.stats = {"transmitted": 0, "delivered": 0, "out_of_range": 0, "lost": 0}

    def attach(self, receiver: RemoteIdReceiver) -> RemoteIdReceiver:
        if receiver.owner_id in self._receivers:
            raise ValueError(f"{receiver.owner_id} already has a receiver")
        self._receivers[receiver.owner_id] = receiver
        return receiver

    def add_link_hook(self, hook, receiver_id: str | None = None) -> None:
        """In-transit interference on delivery to one receiver (or all if None)."""
        self._link_hooks.append((receiver_id, hook))

    def transmit(self, sender_id: str, payloads: list[dict], *, sequence: int,
                 clean: dict | None = None, sender_hooks_changed=()) -> list[dict]:
        """Deliver each message to every other receiver. Returns delivery records."""
        deliveries = []
        for payload in payloads:
            self._tx += 1
            self.stats["transmitted"] += 1
            tx_id = f"tx{self._tx}"
            self._record("rid_transmit", tx_id=tx_id, true_sender=sender_id, sequence=sequence,
                         payload=payload, clean_payload=clean,
                         tampered_by=list(sender_hooks_changed) or None)
            for rid, rx in self._receivers.items():
                if rid == sender_id:
                    continue
                if self._in_range is not None:
                    try:
                        ok, dist = self._in_range(sender_id, rid)
                    except Exception as err:
                        ok, dist = False, None
                        self._record("rid_range_error", tx_id=tx_id, receiver=rid,
                                     error=f"{type(err).__name__}: {err}")
                    if not ok:
                        self.stats["out_of_range"] += 1
                        self._record("rid_out_of_range", tx_id=tx_id, true_sender=sender_id,
                                     receiver=rid, distance_m=None if dist is None else round(dist, 1))
                        continue
                hooks = [h for target, h in self._link_hooks if target in (None, rid)]
                ctx = RidContext(sender_id=sender_id, sequence=sequence, receiver_id=rid)
                got, changed = _apply_hooks(hooks, [copy.deepcopy(payload)], ctx,
                                            lambda h, e, rid=rid: self._record(
                                                "rid_link_error", tx_id=tx_id, receiver=rid,
                                                hook=_hook_name(h), error=f"{type(e).__name__}: {e}"))
                if not got:
                    self.stats["lost"] += 1
                    self._record("rid_lost", tx_id=tx_id, receiver=rid, link_hooks=changed or None)
                linked = getattr(rx, "receive_linked", None)
                for p in got:
                    report = linked(p, sender_id) if linked is not None else rx.receive(p)
                    self.stats["delivered"] += 1
                    d = {"tx_id": tx_id, "true_sender": sender_id, "receiver": rid,
                         "report_id": getattr(report, "observation_id", None),
                         "claimed_uas_id": getattr(report, "claimed_uas_id", None),
                         "link_tampered_by": changed or None}
                    self._record("rid_delivery", **d)
                    deliveries.append(d)
        return deliveries

    def _record(self, kind, **fields):
        if self._recorder is not None:
            self._recorder.record(kind, source="remote_id_bus", **fields)


class RemoteIdBroadcaster:
    """One drone's transmitter. Honest unless given sender hooks (the red team's slot)."""

    def __init__(self, adapter, vehicle_id: str, bus: RemoteIdBus, hooks=(), recorder=None,
                 self_id_text: str = "", uas_id: str | None = None, intent=None, status=None):
        self.vehicle_id = vehicle_id
        self._intent = intent           # () -> target altitude (m) or None: the intent extension
        self._status = status           # () -> mission_state str or None (Phase 3)
        self.uas_id = uas_id or vehicle_id
        self.self_id_text = self_id_text
        self.rate_hz: float | None = None
        self._adapter, self._bus, self._hooks = adapter, bus, list(hooks)
        self._recorder = recorder
        self._seq = 0

    def broadcast_once(self) -> list[dict]:
        self._seq += 1
        try:
            snap = self._adapter.get_snapshot(self.vehicle_id)
        except Exception as err:
            self._record("rid_broadcast_error", sequence=self._seq,
                         error=f"{type(err).__name__}: {err}")
            return []
        target = None
        if self._intent is not None:
            try:
                target = self._intent()
            except Exception:
                target = None
        state = None
        if self._status is not None:
            try:
                state = self._status()
            except Exception:
                state = None
        clean = message_from_snapshot(self.uas_id, snap, self.self_id_text, target, state)
        ctx = RidContext(sender_id=self.vehicle_id, sequence=self._seq)
        sent, changed = _apply_hooks(self._hooks, [copy.deepcopy(clean)], ctx,
                                     lambda h, e: self._record(
                                         "rid_sender_hook_error", sequence=self._seq,
                                         hook=_hook_name(h), error=f"{type(e).__name__}: {e}"))
        if not sent:
            self._record("rid_not_sent", sequence=self._seq, sender_hooks=changed or None)
            return []
        return self._bus.transmit(self.vehicle_id, sent, sequence=self._seq, clean=clean,
                                  sender_hooks_changed=changed)

    async def run(self, rate_hz: float = 1.0, stop: asyncio.Event | None = None) -> None:
        if not rate_hz > 0:
            raise ValueError("rate_hz must be > 0")
        self.rate_hz = rate_hz
        period, next_t = 1.0 / rate_hz, time.monotonic()
        try:
            while stop is None or not stop.is_set():
                self.broadcast_once()
                next_t += period
                delay = max(0.0, next_t - time.monotonic())
                if delay == 0.0:
                    next_t = time.monotonic()
                if stop is None:
                    await asyncio.sleep(delay)
                else:
                    try:
                        async with asyncio.timeout(delay):    # not wait_for: see executor._send
                            await stop.wait()
                    except asyncio.TimeoutError:
                        pass
        finally:
            self.rate_hz = None

    def _record(self, kind, **fields):
        if self._recorder is not None:
            self._recorder.record(kind, vehicle_id=self.vehicle_id, **fields)


# ------------------------------------------------------------ the agent's view
def _r(x, nd=2):
    return round(float(x), nd) if isinstance(x, (int, float)) and math.isfinite(x) else None


def traffic_for_model(receiver: RemoteIdReceiver, now_host_s: float, max_age_s: float = 10.0,
                      max_text_len: int = 200) -> dict:
    """What an agent is told about the other drones: the newest report per claimed id,
    fixed fields only, with its age and trust assessment. Reports older than max_age_s
    are listed as lost contact without a position. The free text is passed through on
    purpose (it is what Remote ID carries) but labeled as unverified."""
    peers, lost, own_claims = [], [], 0
    for claimed, r in sorted(receiver.latest().items()):
        if claimed == receiver.owner_id:
            own_claims += 1                   # someone claims to be us: never "traffic"
            continue
        age = now_host_s - r.received_host_s
        if age > max_age_s:
            lost.append({"uas_id": claimed, "last_heard_s_ago": _r(age, 1)})
            continue
        text = r.self_id_text
        peers.append({
            "uas_id": claimed,
            "report_id": r.observation_id,
            "age_s": _r(age),
            "position": {"north_m": _r(r.num("north_m")), "east_m": _r(r.num("east_m")),
                         "altitude_m": _r(r.num("altitude_m"))},
            "ground_speed_mps": _r(r.num("ground_speed_mps")),
            "track_deg": _r(r.num("track_deg"), 1),
            "vertical_speed_mps": _r(r.num("vertical_speed_mps")),
            "target_altitude_m": _r(r.num("target_altitude_m")),
            "unverified_text": None if text is None else text[:max_text_len],
            "trust_assessment": r.assessment.status.value if r.assessment else None,
        })
    view = {"peers": peers, "lost_contact": lost}
    if own_claims:
        view["ignored_reports_claiming_your_id"] = own_claims
    return view
