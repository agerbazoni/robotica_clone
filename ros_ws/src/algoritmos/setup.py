from setuptools import setup, find_packages

setup(
    name='algoritmos',
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/algoritmos', ['package.xml']),
        ('share/ament_index/resource_index/packages', ['resource/algoritmos']),
    ],
)