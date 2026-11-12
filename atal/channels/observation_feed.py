"""
Per-vehicle observation channel: the ONLY way an agent learns about its drone.

    adapter snapshot (ground truth, simulator side)
        -> strip the adapter's validation verdict          (see atal/core/observation.py)
        -> attack hooks, in order (may modify or drop)      (atal/attacks/hooks.py)
        -> receiver stamps id / channel / sequence / receipt time
        -> trust assessor (sees only what was delivered)    (atal/trust_layer/assessor.py)
        -> authoritative record kept by the feed; the agent gets a deep COPY
        -> recorder (evaluation side): delivered payload + assessment, and, only
           there, which hooks changed it and the clean payload

One feed per vehicle. A feed never serves another vehicle, and an agent holds
only its feed and its action port, never the adapter or a drone handle.

Protected copies and references
    The feed keeps its own copy of every observation it delivered (bounded memory)
    and never hands that copy out: the agent, the assessor, the port and the gate
    each get a deep copy (resolve(observation_id) for the port and gate). A proposal
    therefore only needs the observation_id, which is what an external (LLM /
    tool-call) agent can send; editing any copy cannot change the record, what the
    gate checks, or what later reads return.

Cadence (two different rates, on purpose)
    sample(rate_hz) collects observations at a fixed rate in the background, so
    the assessor sees a regular stream (it gets each observation with the previous
    one) and a short history exists, whatever the agent does. The agent decides at
    its own, slower pace and reads latest() / history() / current() without
    causing new samples. Without a sampler, current() and observe() take one now.
    The evaluator's 20 Hz separation samples are ground truth and are NOT
    observations: no agent or assessor ever sees them.

observe() and current() return None when there is nothing to deliver (adapter
error or a dropped message). Agents must treat None as "no information", not as
"no change".

The source is simulator ground truth (Project AirSim kinematics), named in
`channel`. A sensor-based source (GPS/IMU topics) would be a different channel.
"""
import asyncio
import copy
import time
from collections import OrderedDict, deque
from datetime import datetime, timezone

from atal.attacks.hooks import HookContext
from atal.core.observation import Observation, TrustAssessment, TrustStatus
from atal.trust_layer.assessor import NoTrustAssessor

# Adapter-computed verdict fields: computed on CLEAN data, so never delivered.
VERDICT_FIELDS = {"validation_status", "validation_errors", "validation_warnings"}


def _hook_name(hook) -> str:
    return getattr(hook, "name", None) or getattr(hook, "__name__", None) or type(hook).__name__


def _protected_copy(obs: Observation | None) -> Observation | None:
    return None if obs is None else obs.model_copy(deep=True)


class ObservationFeed:
    def __init__(self, adapter, vehicle_id: str, hooks=(), assessor=None, recorder=None,
                 clock=time.monotonic, remember: int = 1000, history_len: int = 50):
        if vehicle_id not in getattr(adapter, "vehicle_ids", [vehicle_id]):
            raise ValueError(f"unknown vehicle {vehicle_id!r}")
        self.vehicle_id = vehicle_id
        self.channel = f"projectairsim_ground_truth:{vehicle_id}"
        self.clock = clock
        self.sample_hz: float | None = None      # set while sample() runs
        self._adapter = adapter
        self._hooks = list(hooks)
        self._assessor = assessor or NoTrustAssessor()
        self._recorder = recorder
        self._seq = 0
        self._previous: Observation | None = None
        self._issued: OrderedDict[str, Observation] = OrderedDict()
        self._remember = remember
        self._history: deque[Observation] = deque(maxlen=history_len)
        self._new_event: asyncio.Event | None = None
        self._new_loop = None

    # ------------------------------------------------------------ agent side
    def observe(self) -> Observation | None:
        """Take one observation now. Returns the agent's copy."""
        return _protected_copy(self._take())

    def latest(self) -> Observation | None:
        """Most recent delivered observation (agent copy), without sampling."""
        return _protected_copy(self._history[-1]) if self._history else None

    def history(self, n: int | None = None) -> list[Observation]:
        """Up to the last n delivered observations, oldest first (agent copies).
        Dropped messages leave gaps in `sequence`; that is information too."""
        items = list(self._history)
        if n is not None:
            items = items[-n:] if n > 0 else []
        return [_protected_copy(o) for o in items]

    @property
    def last_sequence(self) -> int:
        """Sequence number of the most recent observation attempt (delivered or not)."""
        return self._seq

    async def current(self, after_sequence: int | None = None,
                      timeout_s: float = 2.0) -> Observation | None:
        """An observation taken after sequence `after_sequence` (e.g. the feed's sequence
        when the agent's last command ended, so it reflects the command's outcome).
        Ordering is by sequence, not by clock: on Windows / Python 3.11 time.monotonic
        ticks every ~15 ms, so "received after" by timestamp can be ambiguous.

        With a sampler running, waits for the next sample (up to timeout_s; None if
        none arrives, e.g. every message is being dropped). Without one, observes now.
        """
        if self.sample_hz is None:
            return self.observe()
        deadline = self.clock() + timeout_s
        while True:
            last = self._history[-1] if self._history else None
            if last is not None and (after_sequence is None or last.sequence > after_sequence):
                return _protected_copy(last)
            remaining = deadline - self.clock()
            if remaining <= 0:
                return None
            new = self._new()
            new.clear()
            try:
                async with asyncio.timeout(remaining):     # not wait_for: see executor._send
                    await new.wait()
            except asyncio.TimeoutError:
                return None

    async def sample(self, rate_hz: float, stop: asyncio.Event | None = None) -> None:
        """Observe at a fixed rate until `stop` is set or the task is cancelled."""
        if not rate_hz > 0:
            raise ValueError("rate_hz must be > 0")
        period = 1.0 / rate_hz
        self.sample_hz = rate_hz
        self._record("sampling", status="started", rate_hz=rate_hz)
        try:
            next_t = self.clock()
            while stop is None or not stop.is_set():
                self._take()
                next_t += period
                delay = next_t - self.clock()
                if delay < 0:          # fell behind (slow adapter): don't burst to catch up
                    next_t, delay = self.clock(), 0.0
                if stop is None:
                    await asyncio.sleep(delay)
                else:
                    try:
                        async with asyncio.timeout(delay):  # not wait_for: see executor._send
                            await stop.wait()
                    except asyncio.TimeoutError:
                        pass
        finally:
            self.sample_hz = None
            self._notify()             # wake current() waiters so they fall back / time out
            self._record("sampling", status="stopped", rate_hz=rate_hz, last_sequence=self._seq)

    # ------------------------------------------------------------ port / gate side
    def resolve(self, observation_id) -> Observation | None:
        """A protected copy of an observation this feed delivered and still remembers,
        or None (unknown id, another vehicle's id, forgotten, not a string)."""
        if not isinstance(observation_id, str):
            return None
        return _protected_copy(self._issued.get(observation_id))

    # ------------------------------------------------------------ internal
    def _take(self) -> Observation | None:
        self._seq += 1
        seq = self._seq
        try:
            snap = self._adapter.get_snapshot(self.vehicle_id)
        except Exception as err:
            self._record("observation_error", sequence=seq, error=f"{type(err).__name__}: {err}")
            return None
        received = self.clock()
        source_validation = {
            "status": snap.validation_status.value,
            "errors": list(snap.validation_errors),
        }
        clean = snap.model_dump(mode="json", exclude=VERDICT_FIELDS)

        payload = copy.deepcopy(clean)
        ctx = HookContext(vehicle_id=self.vehicle_id, sequence=seq, received_host_s=received)
        changed_by = []
        for hook in self._hooks:
            before = copy.deepcopy(payload)
            try:
                out = hook(payload, ctx)
            except Exception as err:   # a broken hook is a lost message, never a crash
                self._record("observation_dropped", sequence=seq, hook=_hook_name(hook),
                             reason=f"hook raised {type(err).__name__}: {err}",
                             source_validation=source_validation)
                return None
            if out is None:
                self._record("observation_dropped", sequence=seq, hook=_hook_name(hook),
                             reason="dropped by hook", source_validation=source_validation)
                return None
            if out != before:
                changed_by.append(_hook_name(hook))
            payload = out
        # anything a hook tried to add under a verdict name is not a verdict
        for f in VERDICT_FIELDS & set(payload):
            payload[f"claimed_{f}"] = payload.pop(f)

        obs = Observation(vehicle_id=self.vehicle_id, sequence=seq, channel=self.channel,
                          received_host_s=received, received_at_utc=datetime.now(timezone.utc),
                          payload=copy.deepcopy(payload))
        try:
            assessment = self._assessor.assess(_protected_copy(obs), _protected_copy(self._previous))
        except Exception as err:
            assessment = TrustAssessment(
                observation_id=obs.observation_id, assessor=getattr(self._assessor, "name", "?"),
                assessed=False, status=TrustStatus.NOT_ASSESSED,
                reasons=[f"assessor raised {type(err).__name__}: {err}"])
        obs = obs.model_copy(update={"assessment": assessment})
        self._previous = obs
        self._history.append(obs)
        self._issued[obs.observation_id] = obs
        while len(self._issued) > self._remember:
            self._issued.popitem(last=False)
        self._notify()

        rec = {"observation": obs.model_dump(mode="json"),
               "source_validation": source_validation}   # evaluation side only
        if changed_by:
            rec["tampered_by"] = changed_by
            rec["clean_payload"] = clean
        self._record("observation", **rec)
        return obs

    def _new(self) -> asyncio.Event:
        # created lazily inside the running loop (a feed may outlive one asyncio.run)
        loop = asyncio.get_running_loop()
        if self._new_event is None or self._new_loop is not loop:
            self._new_event, self._new_loop = asyncio.Event(), loop
        return self._new_event

    def _notify(self) -> None:
        if self._new_event is not None:
            self._new_event.set()

    def _record(self, kind, **fields):
        if self._recorder is not None:
            self._recorder.record(kind, vehicle_id=self.vehicle_id, **fields)
