import unittest

from data_preprocessing import data_loader


class TestDataLoaderHelpers(unittest.TestCase):
    def test_get_dataset_primitives_names_valid(self):
        for name in ('LASA', 'LAIR', 'optitrack', 'interpolation', 'joint_space', 'isaac_sim_learned'):
            res = data_loader.get_dataset_primitives_names(name)
            self.assertIsInstance(res, list)

    def test_get_dataset_primitives_names_invalid(self):
        with self.assertRaises(NameError):
            data_loader.get_dataset_primitives_names('UNKNOWN_DATASET')

    def test_select_primitives_str_and_list(self):
        dataset = ['a', 'b', 'c', 'd']
        names, save = data_loader.select_primitives(dataset, '0,2')
        self.assertEqual(names, ['a', 'c'])
        self.assertEqual(save, '0_2')

        names2, save2 = data_loader.select_primitives(dataset, [1, 3])
        self.assertEqual(names2, ['b', 'd'])
        self.assertEqual(save2, '1_3')


if __name__ == '__main__':
    unittest.main()
