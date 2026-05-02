from setuptools import find_packages, setup

package_name = 'ur5_project'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'matplotlib'],
    zip_safe=True,
    maintainer='Daniel',
    maintainer_email='daniel@example.com',
    description='UR5 kinematics and dynamics simulation - BGU 362-1-4231 final project',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Run with: ros2 run ur5_project verify_fk
            'verify_fk = ur5_project.verify_fk:main',
            # Add more entry points here as you build new nodes:
            # 'jacobian_test = ur5_project.jacobian:main',
            # 'trajectory_publisher = ur5_project.trajectory_publisher:main',
        ],
    },
)
