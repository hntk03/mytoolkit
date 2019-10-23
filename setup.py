# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

def main():
    setup(
        name='pytoolkit',
        version='0.0.1',
        zip_safe=False,
        packages=find_packages(),
    )


if __name__ == '__main__':
    main()

