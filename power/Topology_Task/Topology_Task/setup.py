from setuptools import setup, find_packages

setup(
    name='marl2grid',
    version='0.1.0',
    author='',
    author_email='',
    description='A torch modular MARL library for power grids',
    url='',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 1 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.12.2',
    ],
    install_requires=[
        'lightsim2grid==0.9'
    ],
    dependency_links=[
        'git+https://github.com/rte-france/grid2op.git@v1.10.4'  # last grid2op release with duplicated spaces fix
    ]
)