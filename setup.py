from setuptools import find_packages, setup


setup(
    name="sba-uas-reproduction",
    version="0.1.0",
    description="SBA-UAS reproduction with Roach-compatible training and evaluation entry points.",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.7,<3.8",
)
