# pulled from https://github.com/rochacbruno/python-project-template/blob/main/setup.py

from setuptools import find_packages, setup

setup(
    name="dnd_dice_calc",
    version="0.1.0",
    description="A simple UI tool for calculating D&D dice roll probabilities.",
    author="CrepeGoat",
    packages=find_packages(include=["dice", "dice.*"], exclude=["test_*.py"]),
    package_data={"": ["assets/*"]},
    install_requires=["taipy", "numpy"],
    # TODO add entry point for running server
    # entry_points={"console_scripts": ["debug_server = dnd_dice_calc.__main__:main"]},
    extras_require={"test": ["pytest", "hypothesis"]},
)
