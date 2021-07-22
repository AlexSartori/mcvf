import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mcvf",
    version="0.0.1",
    author="Alessandro Sartori",
    author_email="alex.sartori1997@gmail.com",
    description="Motion-Compensated Video Filtering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexsartori/mcvf",
    packages=setuptools.find_packages(),
    package_data={},
    entry_points={
    },
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    keywords='motion-compensated video filtering',
    python_requires='>=3.6',
)
