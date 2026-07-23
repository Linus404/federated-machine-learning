import json
import logging
import unittest

from src.structured_logging import JsonFormatter


class StructuredLoggingTests(unittest.TestCase):
    def test_formatter_emits_consistent_json_context(self) -> None:
        """Preserve application context as machine-readable JSON.

        Returns
        -------
        None
        """
        record = logging.LogRecord(
            "fml.server",
            logging.INFO,
            __file__,
            1,
            "round %s completed",
            (3,),
            None,
        )
        record.context = {"event": "fit_round_completed", "round": 3}

        payload = json.loads(JsonFormatter().format(record))

        self.assertEqual(payload["level"], "INFO")
        self.assertEqual(payload["logger"], "fml.server")
        self.assertEqual(payload["message"], "round 3 completed")
        self.assertEqual(
            payload["context"],
            {"event": "fit_round_completed", "round": 3},
        )
        self.assertTrue(payload["timestamp"].endswith("+00:00"))


if __name__ == "__main__":
    unittest.main()
