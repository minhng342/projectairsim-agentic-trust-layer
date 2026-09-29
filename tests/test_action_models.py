"""ProposedAction / CommandResult contract tests (no simulator)."""
import math

import pytest
from pydantic import ValidationError

from models.action import ActionType, CommandResult, CommandStatus, ProposedAction


def act(action_type, **params):
    return ProposedAction(vehicle_id="Drone1", action_type=action_type, reason="test", **params)


@pytest.mark.parametrize("action_type,params", [
    (ActionType.HOLD, {"duration_s": 3}),
    (ActionType.TAKEOFF, {}),
    (ActionType.ROTATE_TO_HEADING, {"heading_deg": 90}),
    (ActionType.CHANGE_ALTITUDE, {"altitude_m": 10}),
    (ActionType.CHANGE_ALTITUDE, {"altitude_m": 10, "speed_mps": 1.5}),
    (ActionType.MOVE_ALONG_TRACK, {"track_deg": 0, "speed_mps": 3, "duration_s": 5}),
    (ActionType.MOVE_TO_POSITION, {"north_m": -1, "east_m": 8, "altitude_m": 10}),
    (ActionType.MOVE_TO_POSITION, {"north_m": -1, "east_m": 8, "altitude_m": 10, "speed_mps": 4}),
    (ActionType.LAND, {}),
])
def test_valid_actions(action_type, params):
    a = act(action_type, **params)
    assert a.request_id and len(a.request_id) == 32


@pytest.mark.parametrize("action_type,params,message", [
    (ActionType.ROTATE_TO_HEADING, {}, "missing ['heading_deg']"),
    (ActionType.MOVE_ALONG_TRACK, {"track_deg": 0, "speed_mps": 3}, "missing ['duration_s']"),
    # heading on a track move is exactly the ambiguity the review flagged
    (ActionType.MOVE_ALONG_TRACK, {"heading_deg": 0, "speed_mps": 3, "duration_s": 5}, "not allowed"),
    (ActionType.ROTATE_TO_HEADING, {"heading_deg": 90, "speed_mps": 2}, "not allowed"),
    (ActionType.LAND, {"altitude_m": 0}, "not allowed"),
    (ActionType.ROTATE_TO_HEADING, {"heading_deg": 360}, "[0, 360)"),
    (ActionType.MOVE_ALONG_TRACK, {"track_deg": -10, "speed_mps": 3, "duration_s": 5}, "[0, 360)"),
    (ActionType.MOVE_ALONG_TRACK, {"track_deg": 0, "speed_mps": 0, "duration_s": 5}, "> 0"),
    (ActionType.HOLD, {"duration_s": -1}, "> 0"),
    (ActionType.CHANGE_ALTITUDE, {"altitude_m": math.nan}, "finite"),
    (ActionType.CHANGE_ALTITUDE, {"altitude_m": math.inf}, "finite"),
])
def test_invalid_actions(action_type, params, message):
    with pytest.raises(ValidationError) as exc:
        act(action_type, **params)
    assert message in str(exc.value)


def test_reason_is_required():
    with pytest.raises(ValidationError):
        ProposedAction(vehicle_id="Drone1", action_type=ActionType.LAND, reason="")


def test_command_result_round_trips_to_json():
    a = act(ActionType.LAND)
    r = CommandResult(request_id=a.request_id, vehicle_id=a.vehicle_id, action_type=a.action_type,
                      status=CommandStatus.REFUSED, reason="telemetry stale")
    back = CommandResult.model_validate_json(r.model_dump_json())
    assert back.status == CommandStatus.REFUSED and back.sent_to_simulator is False
