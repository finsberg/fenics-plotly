#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

requirements = ["plotly", "fenics", "numpy"]

setup(
    author="Henrik Finsberg",
    author_email="henriknf@simula.no",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="A package for plotting FEniCS objects using plotly",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="fenics_plotly",
    name="fenics_plotly",
    packages=find_packages(include=["fenics_plotly", "fenics_plotly.*"]),
    test_suite="tests",
    url="https://github.com/finsberg/fenics-plotly",
    version="0.1.4",
    project_urls={
        "Documentation": "https://fenics-plotly.readthedocs.io",
        "Source": "https://github.com/finsberg/fenics-plotly",
    },
    zip_safe=False,
)
