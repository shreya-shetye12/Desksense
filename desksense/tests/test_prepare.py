import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from desksense.common import manifest_rows, write_json
from desksense.prepare import audit, prepare_ssv2


class ManifestTests(unittest.TestCase):
    def fixture(self, root):
        rows = []
        for split in ("train", "val", "test"):
            for label in ("a", "b"):
                vid = f"{split}_{label}"
                path = root / (vid + ".mp4")
                path.write_bytes(vid.encode())
                rows.append(dict(id=vid, path=path.name, label=label, split=split, group=vid))
        return rows

    def save(self, root, rows):
        path = root / "manifest.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_valid_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.save(root, self.fixture(root))
            self.assertEqual(audit(path, ["a", "b"])["videos"], 6)
            self.assertTrue(Path(manifest_rows(path)[0]["path"]).is_absolute())

    def test_group_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.fixture(root)
            rows[2]["group"] = rows[0]["group"]
            with self.assertRaisesRegex(ValueError, "group leakage"):
                audit(self.save(root, rows), ["a", "b"])

    def test_exact_duplicate_content_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.fixture(root)
            (root / rows[2]["path"]).write_bytes((root / rows[0]["path"]).read_bytes())
            with self.assertRaisesRegex(ValueError, "duplicate video content"):
                audit(self.save(root, rows), ["a", "b"])

    def test_importer_preserves_official_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            classes = ["Picking something up", "Opening something"]
            train, val = [], []
            for c, label in enumerate(classes):
                for i in range(12):
                    vid = f"train{c}_{i}"
                    (root / (vid + ".webm")).write_bytes(vid.encode())
                    train.append(dict(id=vid, template=label.replace("something", "[something]")))
                for i in range(3):
                    vid = f"holdout{c}_{i}"
                    (root / (vid + ".webm")).write_bytes(vid.encode())
                    val.append(dict(id=vid, template=label.replace("something", "[something]")))
            write_json(root / "train.json", train)
            write_json(root / "val.json", val)
            write_json(root / "classes.json", classes)
            args = SimpleNamespace(videos=str(root), train_json=root / "train.json", val_json=root / "val.json",
                                   classes=root / "classes.json", out=root / "out.csv", seed=42,
                                   max_per_class=0, val_fraction=.2, allow_missing=False)
            prepare_ssv2(args)
            rows = manifest_rows(args.out)
            self.assertTrue(all(r["split"] == "test" for r in rows if r["id"].startswith("holdout")))
            self.assertTrue(all(not r["id"].startswith("holdout") for r in rows if r["split"] != "test"))


if __name__ == "__main__":
    unittest.main()
