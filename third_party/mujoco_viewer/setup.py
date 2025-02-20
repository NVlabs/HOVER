
import os
from setuptools import find_packages, setup

VERSION = '0.1.4'

INSTALL_REQUIRES = (
    ['mujoco >= 2.1.5',
     'glfw >= 2.5.0',
     'imageio',
     'pyyaml']
)

setup(
    name='mujoco-python-viewer',
    version=VERSION,
    author='Rohan P. Singh',
    author_email='rohan565singh@gmail.com',
    url='https://github.com/rohanpsingh/mujoco-python-viewer',
    description='Interactive renderer for MuJoCo Python',
    long_description='Interactive renderer for MuJoCo Python',
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
    ],
    zip_safe=False,
)
