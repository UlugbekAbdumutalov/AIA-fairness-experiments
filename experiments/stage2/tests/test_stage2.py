"""CPU-only synthetic tests. Does not use real authors or download models."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_data import prepare
from common import load_jsonl


class TestPreparation(unittest.TestCase):
    def test_disjoint_balanced_and_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'pan'; (root / 'en').mkdir(parents=True)
            lines = []
            for gender in ('male', 'female'):
                for i in range(3):
                    name = f'{gender}{i}'
                    lines.append(f'{name}:::{gender}')
                    (root / 'en' / f'{name}.xml').write_text(
                        f'<author><documents><document>Example {name}</document>'
                        f'<document>another tweet</document></documents></author>')
            (root / 'en.txt').write_text('\n'.join(lines))
            out = Path(d) / 'out1'
            prepare(root, out, 1, 1, 1, 2026)
            splits = [load_jsonl(out / f'{x}.jsonl') for x in ('train','validation','test')]
            self.assertEqual([len(s) for s in splits], [2,2,2])
            self.assertEqual(len({r['id'] for rows in splits for r in rows}), 6)
            for rows in splits:
                self.assertEqual({r['label'] for r in rows}, {0, 1})
            out2 = Path(d) / 'out2'
            prepare(root, out2, 1, 1, 1, 2026)
            self.assertEqual((out / 'train.jsonl').read_text(), (out2 / 'train.jsonl').read_text())
            with self.assertRaises(FileExistsError):
                prepare(root, out, 1, 1, 1, 2026)


if __name__ == '__main__':
    unittest.main()
