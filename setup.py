from glob import glob
import os

from setuptools import setup

package_name = 'competition_grp5'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rocotics',
    maintainer_email='mads.oritsland@gmail.com',
    description='DAT160 search and rescue competition, group 5',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_controller = competition_grp5.robot_controller:main',
            'marker_detector = competition_grp5.marker_detector:main',
            'coordinator = competition_grp5.coordinator:main',
        ],
    },
)
