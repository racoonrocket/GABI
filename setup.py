import setuptools
print(setuptools.find_packages(where="src"))
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GABI",
    version="0.0.1",
    author="JB-Morlot, J-Mozziconacci, G-Marnier",
    author_email="@mnhn.fr",
    description="Epigenomic profile consolidation using bayesian inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jbmorlot/GABI",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['GABI'],
    package_dir={'GABI': 'src'},
    include_package_data=True,
    package_data ={'GABI': ['src/*.yaml']},
    install_requires=['numpy','scipy','tdqm','pybigwig'],
    python_requires=">=3.6",
)
