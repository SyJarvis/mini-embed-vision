import json
import tempfile
import unittest
from pathlib import Path

from data_generate.translate import load_translated_ids


class TranslateResumeTests(unittest.TestCase):
    def test_skips_only_successful_source_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "translations.jsonl"
            records = [
                {"source_id": "done", "ok": True, "text_en": "source", "text_zh": "译文"},
                {"source_id": "failed", "ok": False, "text_zh": ""},
                {"source_id": "empty", "ok": True, "text_zh": ""},
                {"source_id": "done", "ok": True, "text_en": "source", "text_zh": "译文"},
            ]
            output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))

            self.assertEqual(load_translated_ids(output), {"done"})


if __name__ == "__main__":
    unittest.main()
