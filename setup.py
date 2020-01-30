from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='fireworks-ml',
    version='0.3.9',
    packages=find_packages(),
    author_email="skhan8@mail.einstein.yu.edu",
    description="A batch-processing framework for data analysis and machine learning using PyTorch.",
    long_description=open('README.md').read(),
    url="https://github.com/smk508/fireworks",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
