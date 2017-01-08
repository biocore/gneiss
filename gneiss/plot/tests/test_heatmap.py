from gneiss.plot import heatmap
import pandas as pd
from skbio import TreeNode
import unittest
import os


class HeatmapTest(unittest.TestCase):
    def setUp(self):
        self.fname = 'test.pdf'

    def tearDown(self):
        if os.path.exists(self.fname):
            os.remove(self.fname)

    def test_not_fail(self):
        t = pd.DataFrame({'a': [1, 2, 3],
                          'b': [4, 5, 6],
                          'c': [7, 8, 9]},
                         index=['x', 'y', 'z'])
        r = TreeNode.read([r"((a,b),c);"])
        tr, ts = heatmap(t, r, cmap='viridis', rowlabel_size=14)
        tr.render(file_name=self.fname, tree_style=ts)
        self.assertTrue(os.path.exists(self.fname))
        self.assertTrue(os.path.getsize(self.fname) > 0)

if __name__ == "__main__":
    unittest.main()
