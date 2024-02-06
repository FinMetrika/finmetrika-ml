from setuptools import setup, find_packages

setup(
    author="Ita Cirovic Donev",
    description="Customized machine learning packgae for practical projects.",
    name="finmetrika-ml",
    version="0.1.0",
    packages=find_packages(include=["finmetrika-ml", 
                                    "finmetrika.*"], # include all subpmodules
                           ),
    python_requires=">=3.9"
    install_requires=[
        "pytorch>=2.2.0"
    ]
)