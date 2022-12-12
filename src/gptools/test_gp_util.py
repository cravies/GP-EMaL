import unittest
from gptools.gp_util import human_readable

class TestTikhonovRegularisation(unittest.TestCase):

    def human_readable(self):
        self.assertEqual(human_readable('vsub(vadd(f1,f1),f1)'), 'f1')
        self.assertEqual(human_readable('vadd(f1,f2)'),'f1+f2')
        self.assertEqual(human_readable('vsub(vmul(vadd(f1,f2),f3),f4)'),'(f1+f2)*f3 - f4')

if __name__ == '__main__':
    unittest.main()