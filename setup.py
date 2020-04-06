from setuptools import find_packages, setup

setup(
    name="generation-experiment",
    version="0.0.1",
    description="Generation 셀에서 PoC 및 각종 실험용으로 사용하는 레포지토리입니다.",
    install_requires=[],
    url="https://github.com/scatterlab/generation-experiment.git",
    author="ScatterLab",
    author_email="developers@scatterlab.co.kr",
    packages=find_packages(exclude=["tests"]),
)
