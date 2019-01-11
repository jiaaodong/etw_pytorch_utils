from setuptools import setup, find_packages

setup(
    name='etw_pytorch_utils',
    version='1.0',
    author='Erik Wijmans',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'tqdm', 'visdom'])
