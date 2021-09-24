import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="benchtools",
    version="0.0.1",
    author="Mariana Vivas",
    description="A benchmarking tool for ML classification algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marianaiv/benchmark_clalgoritmos",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "benchtools"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)