import unittest
from gptools.gp_util import human_readable

class TestTikhonovRegularisation(unittest.TestCase):

    def test_human_readable(self):
        print("testing human_readable()")
        self.assertEqual(
            str(human_readable('vsub(vadd(f1,f1),f1)')).replace(" ", ""), 
            'f1'
        )
        self.assertEqual(
            str(human_readable('vadd(f1,f2)')).replace(" ", ""),
            'f1+f2'
        )
        self.assertEqual(
            str(human_readable('vsub(vmul(vadd(f1,f2),f3),f4)')).replace(" ", ""),
            'f3*(f1+f2)-f4'
        )

if __name__ == '__main__':
    print("running.")
    unittest.main()