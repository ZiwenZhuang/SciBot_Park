import setuptools

INSTALL_REQUIRES = [
    # high version will check whether obs is contained by observation_space,
    # which cannot be fully satisfied by pybullet's changeDynamics position
    # limit setting.
    'gym==0.18.0',
    'numpy',
]

setuptools.setup(
    name='scibotpark',
    version='0.0.1',
    packages=setuptools.find_packages(),
    license='MIT License',
    long_description=open('README.md').read(),
    install_requires=INSTALL_REQUIRES,
)