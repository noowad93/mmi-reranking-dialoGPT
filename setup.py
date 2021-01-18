from setuptools import find_packages, setup

setup(
    name="mmi-reranking-dialoGPT",
    version="0.0.1",
    description="dialoGPT 논문에서 언급된 mmi reranking을 구현한 코드입니다.",
    install_requires=[],
    url="https://github.com/noowad93/mmi-reranking-dialogpt.git",
    author="dawoon jung",
    author_email="dawoon@scatterlab.co.kr",
    packages=find_packages(exclude=["tests"]),
)
