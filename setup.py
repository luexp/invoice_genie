from setuptools import setup, find_packages

"""
Setup script for the API package.

This script uses setuptools to configure and install the API package.
It defines the package name and automatically discovers and includes all packages in the project.
"""

setup(
    name="API",
    packages=find_packages(),
    description="API package for image processing and text extraction",
    long_description="A comprehensive API package that includes functionality for image processing, "
                     "OCR, and machine learning model integration.",
    version="0.1.0",
    authors="Victor, Luigi"
)
