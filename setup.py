from setuptools import setup

setup(
    name='redphot',
    version='0.1',
    author='Sebastian Gomez',
    author_email='sebastian.gomez@austin.utexas.edu',
    description='Functions to reduce photometry.',
    url='https://github.com/gmzsebastian/redphot',
    license='MIT License',
    python_requires='>=3.6',
    packages=['redphot'],
    include_package_data=True,
    package_data={'redphot': ['ref_data/*']},
    install_requires=[
        'numpy',
        'matplotlib',
        'astropy',
        'scipy',
        'emcee'
    ]
)
