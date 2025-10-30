from setuptools import setup, find_packages

setup(
    name="PHUMA",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)