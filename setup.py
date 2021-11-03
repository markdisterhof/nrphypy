from setuptools import setup

setup(
    name='nrphypy',
    version='0.9.0',    
    description='Python module for 5G NR sync signals and decoding',
    url='https://github.com/markdisterhof/nrphypy',
    author='Mark Disterhof',
    author_email='mardis@uni-bremen.de',
    license='GNU General Public License v3.0',
    packages=['nrphypy'],
    install_requires=['numpy'],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Telecommunications Industry',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
