from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="muben",
    version="0.0.1",
    author="Yinghao Li",
    author_email="yinghaoli@gatech.edu",
    license="MIT",
    url="https://github.com/Yinghao-Li/MUBen",
    description="Benchmark for molecular uncertainty estimation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="machine-learning uncertainty-estimation materials-science materials-property-prediction",
    zip_safe=False,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Text Processing",
        "Topic :: Text Processing :: Linguistic",
    ],
    packages=find_packages(),
    python_requires=">=3.9",
)
