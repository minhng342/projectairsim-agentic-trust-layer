"""A drone's two ends of the peer-message link (Phase 3).

    SwarmReceiver   on the bus: Remote ID position reports go to the drone's
                    RemoteIdReceiver; typed peer messages (atal/core/peer_messages.py) are
                    validated (invalid ones dropped and recorded, never seen by anything
                    else) and handed to the drone's coordinator through on_message
    PeerMessenger   this drone's transmitter for peer messages; sender hooks are the
                    insider's slot in Phase 4 (honest = none)

Transport only: what a message means is decided in atal/agents/team_coordinator.py.
"""
import copy

from atal.channels.radio import RidContext, apply_hooks, hook_name
from atal.core.peer_messages import PEER_MESSAGE, is_peer_message


class Delivered:
    """What the bus records about a delivered peer message (its id and claimed sender)."""

    def __init__(self, observation_id, claimed_uas_id):
        self.observation_id, self.claimed_uas_id = observation_id, claimed_uas_id


class SwarmReceiver:
    """authenticated=True: the bus tells this receiver who really transmitted
    (receive_linked); a message claiming another identity is dropped before anything
    reads it (identity_mismatch). False: the claimed id is believed (spoofing condition)."""

    def __init__(self, owner_id: str, rid_receiver, on_message, sim_clock, recorder=None,
                 authenticated: bool = True):
        self.owner_id = owner_id
        self.rid = rid_receiver
        self._on_message, self._sim_clock, self._recorder = on_message, sim_clock, recorder
        self.authenticated = authenticated
        self.dropped = 0
        self.identity_mismatches = 0
        self.processing_errors = 0

    def receive_linked(self, payload, link_sender: str):
        if self.authenticated:
            peer = is_peer_message(payload)
            claimed = (payload.get("sender_id") if peer else payload.get("uas_id")) \
                if isinstance(payload, dict) else None
            if claimed != link_sender:
                self.identity_mismatches += 1
                self._record("identity_mismatch", link_sender=link_sender, claimed=claimed,
                             message_kind="peer_message" if peer else "remote_id",
                             msg_id=payload.get("msg_id") if isinstance(payload, dict) else None)
                return None
        return self.receive(payload)

    def receive(self, payload):
        if not is_peer_message(payload):
            return self.rid.receive(payload)
        try:
            msg = PEER_MESSAGE.validate_python(copy.deepcopy(payload))
        except Exception as err:
            self.dropped += 1
            errors = getattr(err, "errors", None)
            self._record("peer_message_dropped", payload=payload,
                         errors=[f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
                                 for e in errors()][:10] if callable(errors) else str(err)[:300])
            return None
        now = self._sim_clock()
        self._record("peer_message", msg=msg.model_dump(mode="json"), received_sim_s=now)
        try:
            self._on_message(msg, now)
        except Exception as err:
            # delivered on the radio, but this drone failed to act on it: an implementation
            # failure, counted and failing the run (never mistaken for attack resistance)
            self.processing_errors += 1
            self._record("peer_message_error", msg_id=msg.msg_id,
                         error=f"{type(err).__name__}: {err}")
        return Delivered(msg.msg_id, msg.sender_id)

    def _record(self, kind, **fields):
        if self._recorder is not None:
            self._recorder.record(kind, vehicle_id=self.owner_id, source="swarm", **fields)


class PeerMessenger:
    """Sends this drone's peer messages on the bus. Honest unless given sender hooks."""

    def __init__(self, vehicle_id: str, bus, hooks=(), recorder=None):
        self.vehicle_id, self._bus, self._hooks = vehicle_id, bus, list(hooks)
        self._recorder = recorder
        self._seq = 0
        self.sent = 0

    def send(self, msg) -> list[dict]:
        self._seq += 1
        clean = msg.model_dump(mode="json")
        ctx = RidContext(sender_id=self.vehicle_id, sequence=self._seq)
        sent, changed = apply_hooks(self._hooks, [copy.deepcopy(clean)], ctx,
                                     lambda h, e: self._record(
                                         "peer_sender_hook_error", hook=hook_name(h),
                                         error=f"{type(e).__name__}: {e}"))
        self.sent += 1
        self._record("peer_message_sent", msg=clean, tampered_by=changed or None)
        if not sent:
            return []
        return self._bus.transmit(self.vehicle_id, sent, sequence=self._seq, clean=clean,
                                  sender_hooks_changed=changed)

    def _record(self, kind, **fields):
        if self._recorder is not None:
            self._recorder.record(kind, vehicle_id=self.vehicle_id, source="swarm", **fields)
