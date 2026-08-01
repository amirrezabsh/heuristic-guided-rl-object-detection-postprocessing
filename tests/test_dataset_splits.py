import random
import tempfile
import unittest
from pathlib import Path

from scripts.create_coco_subsets import sample_images, write_paths


class DatasetSplitTest(unittest.TestCase):
    def test_sampling_is_deterministic_and_unique(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_dir = Path(tmp_dir)
            for index in range(10):
                (image_dir / f"{index:04d}.jpg").write_bytes(b"")

            first = sample_images(image_dir, 5, random.Random(7))
            second = sample_images(image_dir, 5, random.Random(7))

            self.assertEqual(first, second)
            self.assertEqual(5, len(set(first)))

    def test_write_paths_uses_absolute_sorted_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            images = [root / "b.jpg", root / "a.jpg"]
            output = root / "split.txt"

            write_paths(output, images)

            self.assertEqual(
                [str((root / "a.jpg").resolve()), str((root / "b.jpg").resolve())],
                output.read_text(encoding="utf-8").splitlines(),
            )


if __name__ == "__main__":
    unittest.main()
