# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 13:34:15 2025

@author: rohdo
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rsjetstruct",
    version="1.0",
    author="Coleman Rohde",
    author_email="rohdog2003@gmail.com",
    description="",
    long_description = long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rohdog2003/rsjetstruct",
    packages=setuptools.find_packages(),
    license="GNU GPL v3",
    python_requires='>=3.8',
)