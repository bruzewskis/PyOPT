import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyOPT", # Replace with your own username
    version="0.0.1",
    author="Seth Bruzewski",
    author_email="bruzewskis@unm.edu",
    description="A radio astronomy scheduling package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bruzewskis/PyOPT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)