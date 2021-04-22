from setuptools import find_packages, setup

requirements = [
    "numpy>=1.16",
    "torch",
    "wbml>=0.3",
    "plum-dispatch>=1",
    "backends>=1",
    "backends-matrix>=1",
    "stheno>=1",
    "varz>=0.6",
]

setup(
    packages=find_packages(exclude=["docs"]),
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
)
