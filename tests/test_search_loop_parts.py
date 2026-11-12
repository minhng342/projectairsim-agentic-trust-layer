"""The search loop's two plug-in parts: any chooser goes with any role, and each combination
is logged under the planner name the run logs used before the split."""
import pytest

from atal.agents.fixed_chooser import FixedChooser
from atal.agents.llm_chooser import LLMChooser
from atal.agents.search_loop import SearchLoop, SoloRole
from atal.agents.team_coordinator import TeamRole
from atal.core.belief import LocalBelief
from atal.core.mission import Mission
from atal.trust_layer.invariant_guard import SarGuard
from tests.test_sar_phase2 import MISSION


class _Stub:
    """Stands in for the feed, port and belief; the loop does not touch them until run()."""


def _loop(mission, chooser, role=None, guard=None):
    return SearchLoop("Drone1", mission, _Stub(), _Stub(), _Stub(), chooser=chooser,
                      role=role, guard=guard)


@pytest.fixture
def mission():
    return Mission.model_validate(MISSION)


def test_every_chooser_and_role_has_its_logged_name(mission):
    guard = SarGuard(mission, LocalBelief("Drone1", mission), (0.0, 0.0))
    assert _loop(mission, FixedChooser()).name == "FixedSectorPlanner"
    llm_solo = _loop(mission, LLMChooser(model="test"), guard=guard)
    assert llm_solo.name == "LLMSectorPlanner"
    assert _loop(mission, FixedChooser(), TeamRole(None)).name == "SwarmFixedPlanner"
    llm_team = _loop(mission, LLMChooser(model="test"), TeamRole(None), guard=guard)
    assert llm_team.name == "SwarmLLMPlanner"
    # the team part of the system prompt is there only for a team drone
    assert "one drone of a team" in str(llm_team.chooser._agent._instructions)
    assert "one drone of a team" not in str(llm_solo.chooser._agent._instructions)


def test_parts_are_bound_to_their_loop(mission):
    chooser, role = FixedChooser(), SoloRole()
    loop = _loop(mission, chooser, role)
    assert chooser.loop is loop and role.loop is loop and loop.role is role


def test_the_llm_chooser_needs_the_guard(mission):
    with pytest.raises(ValueError, match="guard"):
        _loop(mission, LLMChooser(model="test"))
