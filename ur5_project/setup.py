from setuptools import find_packages, setup

package_name = 'ur5_project'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Daniel and Itai',
    maintainer_email='zionidan@post.bgu.ac.il',
    description='UR5 kinematics, dynamics, and trajectory simulation '
                '(BGU course 362-1-4231 final project)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'workspace          = ur5_project.workspace:main',
            'jacobian           = ur5_project.jacobian:main',
            'simulation         = ur5_project.simulation:main',
            'simulation_gazebo  = ur5_project.simulation_gazebo:main',
            'verify_fk          = ur5_project.verify_fk:main',
            'test_trajectory    = ur5_project.test_trajectory:main',
            'trajectory         = ur5_project.trajectory:main',
        ],
    },
)
