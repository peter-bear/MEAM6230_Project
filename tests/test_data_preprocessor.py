import unittest

import numpy as np

from data_preprocessing.data_preprocessor import DataPreprocessor


class DummyParams:
    trajectories_resample_length = 10
    state_increment = 0.1
    workspace_dimensions = 2
    dynamical_system_order = 1
    workspace_boundaries_type = 'from data'
    workspace_boundaries = [[0, 1], [0, 1]]
    evaluation_samples_length = 5
    dataset_name = 'LASA'
    selected_primitives_ids = '0'
    spline_sample_type = 'from data'
    imitation_window_size = 2


class TestDataPreprocessorHelpers(unittest.TestCase):
    def setUp(self):
        self.params = DummyParams()
        self.dp = DataPreprocessor(self.params, verbose=False)

    def test_workspace_boundaries_from_data(self):
        demo1 = np.array([[0, 1, 2], [0, 1, 2]])
        demo2 = np.array([[-1, 0, 1], [-2, -1, 0]])
        x_min, x_max = self.dp.get_workspace_boundaries([demo1, demo2])
        # expected orig min and max
        np.testing.assert_allclose(x_max, np.array([2.15, 2.2]), rtol=1e-6)
        np.testing.assert_allclose(x_min, np.array([-1.15, -2.2]), rtol=1e-6)

    def test_get_trajectories_length_and_eval_indexes(self):
        demo1 = np.array([[0, 1, 2], [0, 1, 2]])
        demo2 = np.array([[0, 1, 2], [0, 1, 2]])
        max_len, lengths, eval_indexes = self.dp.get_trajectories_length([demo1, demo2], 2)
        self.assertEqual(max_len, 3)
        self.assertEqual(lengths, [3, 3])
        self.assertTrue(all(np.array_equal(idx, np.array([0, 1, 2])) for idx in eval_indexes))

    def test_get_goals(self):
        demo1 = np.array([[0, 1, 2], [0, 1, 2]])
        demo2 = np.array([[-1, 0, 1], [-2, -1, 0]])
        primitive_ids = np.array([0, 1])
        goals = self.dp.get_goals([demo1, demo2], primitive_ids)
        expected = np.array([[2, 2], [1, 0]])
        np.testing.assert_allclose(goals, expected)


if __name__ == '__main__':
    unittest.main()
