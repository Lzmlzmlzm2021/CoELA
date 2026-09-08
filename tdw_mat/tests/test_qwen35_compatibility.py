import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


TDW_MAT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TDW_MAT_ROOT))

from LLM.LLM import LLM


class Qwen35CompatibilityTests(unittest.TestCase):
    def _make_llm(self):
        sampling = SimpleNamespace(
            max_tokens=256,
            t=0.7,
            top_p=1.0,
            n=1,
            debug=False,
            logprobs=0,
            echo=False,
        )
        return LLM(
            source="openai",
            lm_id="Qwen3.5-4B",
            prompt_template_path=str(TDW_MAT_ROOT / "LLM" / "prompt_single.csv"),
            communication=False,
            cot=True,
            sampling_parameters=sampling,
            agent_id=0,
        )

    def test_native_thinking_is_disabled_only_when_opted_in(self):
        with patch.dict(os.environ, {
                "OPENAI_BASE_URL": "http://127.0.0.1:1/v1",
                "OPENAI_API_KEY": "test",
                "TDW_MAT_DISABLE_MODEL_THINKING": "1",
        }, clear=False):
            llm = self._make_llm()
        self.assertEqual(
            llm.sampling_params["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_native_thinking_default_is_unchanged(self):
        with patch.dict(os.environ, {
                "OPENAI_BASE_URL": "http://127.0.0.1:1/v1",
                "OPENAI_API_KEY": "test",
        }, clear=False):
            os.environ.pop("TDW_MAT_DISABLE_MODEL_THINKING", None)
            llm = self._make_llm()
        self.assertNotIn("extra_body", llm.sampling_params)


if __name__ == "__main__":
    unittest.main()
