import tempfile
import unittest
from pathlib import Path
from localai import knowledge

class KnowledgeTests(unittest.TestCase):
    def test_html_extraction_removes_scripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "page.html"
            p.write_text("<h1>Study notes</h1><script>secret()</script><p>Python uses indentation.</p>", encoding="utf-8")
            text = knowledge.extract_text(p)
            self.assertIn("Study notes", text)
            self.assertIn("Python uses indentation", text)
            self.assertNotIn("secret()", text)

    def test_chunk_text_has_overlap_and_no_empty_chunks(self):
        text = ("Local neural network training uses diverse examples. " * 150)
        chunks = knowledge._chunk_text(text, max_chars=400, overlap=50)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(chunk.strip() for chunk in chunks))

    def test_index_search_and_remove(self):
        old_index = knowledge.load_index()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ai_notes.txt"
            path.write_text(("Transformers use attention to model token relationships. " * 24), encoding="utf-8")
            result = knowledge.index_files([path])
            self.assertEqual(len(result["success"]), 1)
            found = knowledge.search("transformers attention", limit=3)
            self.assertTrue(found)
            self.assertEqual(found[0]["source"], "ai_notes.txt")
            knowledge.remove_source(str(path.resolve()))
        # Restore the user's existing index so the test is non-destructive.
        knowledge.save_index(old_index)

if __name__ == "__main__":
    unittest.main()
