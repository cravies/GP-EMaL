import unittest
from sympy import sympify
from gptools.gp_util import human_readable, grad_tree
import gpmalmo.rundata as rd

class TestTikhonovRegularisation(unittest.TestCase):

    def test_human_readable(self):
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

    def test_grad_tree(self):
        #set number of features for each test using rd.num_features
        #(this is our runtime settings file located in gpmalmo)
        rd.num_features=2
        expr1=sympify('f0*f1')
        self.assertEqual(
            grad_tree(expr1),
            [sympify('f1'),sympify('f0')]
        )
        rd.num_features=3
        expr2=sympify('f0*f1*f2 + f1**2')
        self.assertEqual(
            grad_tree(expr2),
            [sympify('f1*f2'),
            sympify('f0*f2+2*f1'),
            sympify('f0*f1')]
        )
        rd.num_features=4
        expr3=sympify('f1*f3 + f0*f1 + 5*f0 - 8*f2')
        self.assertEqual(
            grad_tree(expr3),
            [sympify('f1+5'),
            sympify('f3+f0'),
            sympify('-8'),
            sympify('f1')]
        )

if __name__ == '__main__':
    print("Running tests.")
    unittest.main()