import unittest
from unittest.mock import patch

from transport_challenge_multi_agent.transport_challenge import (
    _persist_generated_count_and_position,
)


class DatasetImmutabilityTests(unittest.TestCase):
    def test_official_metadata_path_is_never_opened_for_writing(self):
        with patch("builtins.open") as mocked_open:
            persisted = _persist_generated_count_and_position(
                "dataset/dataset_test/5a_0_0_metadata.json",
                {"0": {"x": 0, "y": 0, "z": 0}},
                enabled=False,
            )

        self.assertFalse(persisted)
        mocked_open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
