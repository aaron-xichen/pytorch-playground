from setuptools import setup, find_packages

with open("requirements.txt") as requirements_file:
    REQUIREMENTS = requirements_file.readlines()

setup(
    name="pytorch-playground",
    version="1.0.0",
    author='Aaron Chen',
    author_email='aaron.xichen@gmail.com',
    packages=find_packages(),
    entry_points = {
        'console_scripts': [
            'quantize=quantize:main',
        ]
    },
    install_requires=REQUIREMENTS,

)
