import unittest

from src.rlnas_od.search_space import YoloNeckSearchSpace


class YoloNeckSearchSpaceTest(unittest.TestCase):
    def test_search_has_four_required_stages(self):
        space = YoloNeckSearchSpace(max_blocks=4)
        state = space.initial_state()

        for _ in range(4):
            actions = space.valid_actions(state)
            self.assertEqual(2, len(actions))
            self.assertNotIn("stop", [action.name for action in actions])
            state = space.transition(state, actions[0])

        self.assertEqual(4, state["position"])
        actions = space.valid_actions(state)
        self.assertEqual(["stop"], [action.name for action in actions])

    def test_actions_only_toggle_shortcut(self):
        space = YoloNeckSearchSpace(max_blocks=4)
        actions = {action.name: action for action in space.valid_actions(space.initial_state())}

        action = actions["s1"]

        self.assertEqual(1.0, action.width_mult)
        self.assertEqual(1, action.repeats)
        self.assertTrue(action.shortcut)

    def test_requires_exactly_four_stages(self):
        with self.assertRaises(ValueError):
            YoloNeckSearchSpace(max_blocks=3)


if __name__ == "__main__":
    unittest.main()
