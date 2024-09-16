from setuptools import setup, find_packages

setup(
    name='pipeline',
    version='0.1',
    packages=find_packages(),
    # include any dependencies here
    install_requires=[
        'banditpam',
        'scanpy',
        'leidenalg',
    ],
)