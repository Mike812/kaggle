from setuptools import setup

setup(
   name='alternate_dimension',
   version='1.0',
   description='Predict alternate dimension transport',
   author='Michael Krug',
   author_email='michaelkrug92@gmail.com',
   packages=['alternate_dimension'],
   install_requires=['pandas', 'numpy', 'scikit-learn', 'xgboost', 'pytest'],
)