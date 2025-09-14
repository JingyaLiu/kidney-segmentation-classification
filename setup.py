#!/usr/bin/env python3
"""
Setup script for nnUNet
Installs the package and sets up the environment
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nnunet",
    version="1.0.0",
    author="nnUNet Team",
    author_email="contact@nnunet.org",
    description="Neural Network for Medical Image Segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MIC-DKFZ/nnUNet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "nnUNet_predict=nnUNet_predict:main",
            "nnUNet_ensemble=nnUNet_ensemble:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.py", "*.md", "*.txt"],
    },
)
