import unittest
from trees import _search_tree_idx

class TestSearchTreeIdxFunction(unittest.TestCase):
    def test_1(self):
        # Test case 1
        sequence_1 = [100, 250, 350]
        name_1 = 120
        self.assertEqual(_search_tree_idx(name_1, sequence_1), 1)

    def test_2(self):
        # Test case 2
        sequence_2 = [50, 75, 100, 125]
        name_2 = 80
        self.assertEqual(_search_tree_idx(name_2, sequence_2), 2)

    def test_3(self):
        # Test case 3
        sequence_3 = [10, 20, 30]
        name_3 = 15
        self.assertEqual(_search_tree_idx(name_3, sequence_3), 1)

    def test_4(self):
        # Test case 4
        sequence_4 = [5, 10, 15]
        name_4 = 12
        self.assertEqual(_search_tree_idx(name_4, sequence_4), 2)

    def test_5(self):
        # Test case 5
        sequence_5 = [1, 2, 3]
        name_5 = 2
        self.assertEqual(_search_tree_idx(name_5, sequence_5), 2)

    def test_6(self):
        # Test case 6
        sequence_6 = [1, 2, 3]
        name_6 = 4
        self.assertEqual(_search_tree_idx(name_6, sequence_6), None)

    def test_7(self):
        # Test case 7
        sequence_7 = [100, 250, 600]
        name_7 = 0
        self.assertEqual(_search_tree_idx(name_7, sequence_7), 0)

    def test_8(self):
        # Test case 8
        sequence_7 = [100, 250, 600]
        name_7 = 99
        self.assertEqual(_search_tree_idx(name_7, sequence_7), 0)

    def test_9(self):
        # Test case 9
        sequence_7 = [100, 250, 600]
        name_7 = 100
        self.assertEqual(_search_tree_idx(name_7, sequence_7), 1)

    def test_10(self):
        # Test case 10
        sequence_7 = [100, 250, 600]
        name_7 = 101
        self.assertEqual(_search_tree_idx(name_7, sequence_7), 1)


if __name__ == '__main__':
    unittest.main()
