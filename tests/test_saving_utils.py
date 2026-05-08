import tempfile
import unittest
from pathlib import Path

from evaluation.utils.saving import save_best_stats_txt, save_stats_txt


class TestSavingUtils(unittest.TestCase):
    def test_save_best_stats_txt_creates_expected_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_best_stats_txt(
                save_path=tmp_dir,
                best_n_spurious=3,
                best_metric=1.23,
                best_RMSE=0.1,
                best_DTWD=0.2,
                best_FD=0.3,
                best_DTWD_std=0.01,
                gpu_status='Using device: cpu\n',
                i=42,
            )

            content = (Path(tmp_dir) / 'best_model.txt').read_text(encoding='utf-8')
            self.assertIn('Number of unsuccessful trajectories: 3', content)
            self.assertIn('RMSE + DTWD + FD: 1.23', content)
            self.assertIn('Iteration number: 42', content)
            self.assertIn('Using device: cpu', content)

    def test_save_stats_txt_appends_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_stats_txt(
                save_path=tmp_dir,
                best_n_spurious=2,
                best_metric=9.87,
                best_RMSE=0.4,
                best_DTWD=0.5,
                best_FD=0.6,
                i=7,
            )

            content = (Path(tmp_dir) / 'training_evaluation_summary.txt').read_text(encoding='utf-8')
            self.assertIn('Iteration number: 7', content)
            self.assertIn('Number of unsuccessful trajectories: 2', content)
            self.assertIn('RMSE + DTWD + FD: 9.87', content)


if __name__ == '__main__':
    unittest.main()
