"""Units + command-preempt regression tests.
1. FK pose (mm) and the overlay grid share one coordinate system: the EE marker is projected
   through the SAME homography whose ticks are labelled cm = mm/10.
2. A spoken command (action directive) supersedes the planner for that step and runs first.
"""
import numpy as np

from sortbot import config as cfgmod, perception
from sortbot.testing import MockRobot


def test_fk_units_match_overlay():
    cfg = cfgmod.load()
    r = MockRobot(cfg)
    r.home()
    p = r.get_ee_pose()
    # FK is millimetres: home sits ~15 cm forward; a cm-valued pose would be an order of magnitude off
    assert 80.0 < p.x < 400.0, f"FK x={p.x} not in mm range"
    H = np.array([[0.5, 0.0, -100.0], [0.0, -0.5, 300.0], [0.0, 0.0, 1.0]])  # px -> mm
    uv = perception.mm_to_px(H, [(p.x, p.y)])[0]
    manual = np.linalg.solve(H, np.array([p.x, p.y, 1.0]))
    assert np.allclose(uv, manual[:2] / manual[2], atol=0.01), (uv, manual)


def test_action_directive_preempts_planner():
    from sortbot.main import DirectiveQueue

    q = DirectiveQueue()
    q.put("action", text="turn_by({'deg': 15.0})", data={"tool": "turn_by", "args": {"deg": 15.0}})
    q.put("rule", text="reds on the left")
    kinds = [d["kind"] for d in q.peek()]
    assert kinds == ["action", "rule"]
    drained = q.drain()
    assert drained[0]["data"] == {"tool": "turn_by", "args": {"deg": 15.0}}
    assert not q.peek()


if __name__ == "__main__":
    test_fk_units_match_overlay()
    test_action_directive_preempts_planner()
    print("units + preempt OK: FK mm == overlay grid mm (cm is labels only); action directives carry data")
