from setuptools import setup, find_packages

setup(
    author="Ita Cirovic Donev",
    description="Customized machine learning package for practical projects.",
    name="finmetrika-ml",
    version="0.1.1",
    packages=find_packages(include=["finmetrika_ml", 
                                    "finmetrika.*"], # include all subpmodules
                           ),
    python_requires=">=3.9",
    #install_requires=[
    #    "pytorch>=2.0.1"
    #]
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Financial and Insurance Industry",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
    ]
)