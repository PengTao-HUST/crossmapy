import os
from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="crossmapy",
    version="0.0.1",
    author="Peng Tao",
    author_email="taopeng543@gmail.com",
    url='https://github.com/PengTao-HUST/crossmapy',
    description="causal inference under dynamical causality framework",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'scipy',
        'scikit-learn',
        'networkx',
        'numba',
        'seaborn',
        'matplotlib',
        'numpy'
    ],
    license="MIT Licence",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ]
)
