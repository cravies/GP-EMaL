import unittest
from sympy import sympify
from gptools.gp_util import human_readable, grad_tree, functional_complexity
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
        self.assertEqual(grad_tree(expr1),
            [sympify('f1'),sympify('f0')]
        )
        rd.num_features=3
        expr2=sympify('f0*f1*f2 + f1**2')
        self.assertEqual(grad_tree(expr2),
            [sympify('f1*f2'),sympify('f0*f2+2*f1'),sympify('f0*f1')]
        )
        rd.num_features=4
        expr3=sympify('f1*f3 + f0*f1 + 5*f0 - 8*f2')
        self.assertEqual(grad_tree(expr3),
            [sympify('f1+5'),sympify('f3+f0'),sympify('-8'),sympify('f1')]
        )

class TestFunctional(unittest.TestCase):
    """
    complexity dictionary:
    'vadd':1,'vsub':1,'vmul':1,'vdiv':1, 'max':2,
    'min':2,'np_if':2,'sigmoid':2,'relu':2,'abs':2,'
    """
    def test_functional_complexity(self):
        self.assertEqual(
            functional_complexity('vsub(vadd(f1,f2),f3)'),2
        )
        self.assertEqual(
            functional_complexity('relu(vmul(vdiv(f1,f3),max(f2,f4)))'),6
        )
        self.assertEqual(
            functional_complexity('sigmoid(relu(abs(f1)))'),6
        )

if __name__ == '__main__':
    print("Running tests.")
    unittest.main()