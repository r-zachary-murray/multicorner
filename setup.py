from setuptools import setup

setup(
    name="multicorner",  # Replace with actual package name
    version="1.0.0",
    author="Zachary Murray",
    author_email="zachary.murray@geoazur.unice.fr",
    description="A small module to produce corner-plots of multimodal distributions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/r-zachary-murray/multicorner",  # GitHub or project link
    package_dir={"": "src"},  # Point to src/
    py_modules=["multicorner"],  # Because you're using a single file
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.3",
        "scipy>=1.10.1",
        "scikit-learn>=1.6.0"
    ],
    classifiers=[
        "Intended Audience :: Researchers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)

