#!/usr/bin/env python
from setuptools import find_packages, setup

requirements = []
with open("requirements.txt") as f:
    for line in f:
        stripped = line.split("#")[0].strip()
        if len(stripped) > 0:
            requirements.append(stripped)

setup(
    name="scivision_test_plugin",
    version="0.0.1",
    description="scivision test plugin",
    author="Alan R. Lowe",
    author_email="alowe@turing.ac.uk",
    url="https://github.com/quantumjot/scivision-test-plugin",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.9",
)
