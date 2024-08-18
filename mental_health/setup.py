from setuptools import setup

setup(
   name='mental_health',
   version='1.0',
   description='Predict mental health status',
   author='Michael Krug',
   author_email='michaelkrug92@gmail.com',
   packages=['mental_health'],
   install_requires=['pandas', 'numpy', 'scikit-learn', 'xgboost', 'pytest'],
)