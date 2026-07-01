from setuptools import find_packages, setup

package_name = 'partec'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='abril',
    maintainer_email='agerbazoni@udesa.edu.ar',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
	entry_points={
        'console_scripts': [
            'vision = partec.vision_node:main',
            'cerebro = partec.state_machine:main'
        ],
    },
)
