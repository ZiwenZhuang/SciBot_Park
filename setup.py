import setuptools

INSTALL_REQUIRES = [
    'gym',
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