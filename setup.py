# -*- coding: utf-8 -*-

# Learn more: https://github.com/lukeparry/pyocl/setup.py

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pyocl',
    version='0.1.0',
    description='Helper classes for running OpenCL with Python based on PyOpencl',
    long_description=readme,
    author='Luke Parry',
    author_email='me@kennethreitz.com',
    url='https://github.com/drlukeparry/pyocl',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

