from gneiss.plot import diamondtree
from skbio import TreeNode
from skbio.util import get_data_path
import unittest
import os


class BlobTreeTest(unittest.TestCase):
    def setUp(self):
        self.fname = 'test.pdf'
        self.newick = 'example.nwk'

    def tearDown(self):
        if os.path.exists(self.fname):
            os.remove(self.fname)

    def test_not_fail(self):
        st = TreeNode.read(get_data_path(self.newick))
        tr, ts = diamondtree(st, collapsed_nodes=['y5', 'y18'],
                             breadth_scaling=6, depth_scaling=30,
                             cladecolors={'y5': '#FF0000', 'y18': '#0000FF'},
                             bgcolors={'y29': '#00FF00'})

        tr.render(file_name=self.fname, tree_style=ts)
        self.assertTrue(os.path.exists(self.fname))
        self.assertTrue(os.path.getsize(self.fname) > 0)

    def test_no_collapsed_nodes(self):
        st = TreeNode.read(get_data_path(self.newick))
        tr, ts = diamondtree(st, breadth_scaling=6, depth_scaling=30,
                             cladecolors={'y5': '#FF0000', 'y18': '#0000FF'},
                             bgcolors={'y29': '#00FF00'})

        tr.render(file_name=self.fname, tree_style=ts)
        self.assertTrue(os.path.exists(self.fname))
        self.assertTrue(os.path.getsize(self.fname) > 0)

    def test_no_colors(self):
        st = TreeNode.read(get_data_path(self.newick))
        tr, ts = diamondtree(st, breadth_scaling=6, depth_scaling=30)

        tr.render(file_name=self.fname, tree_style=ts)
        self.assertTrue(os.path.exists(self.fname))
        self.assertTrue(os.path.getsize(self.fname) > 0)


if __name__ == "__main__":
    unittest.main()
