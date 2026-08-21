from setuptools import setup

setup(
    name='redphot',
    version='0.1.0',
    author='Sebastian Gomez',
    author_email='sebastian.gomez@austin.utexas.edu',
    description='Robust time-domain optical photometry of supernovae.',
    url='https://github.com/gmzsebastian/redphot',
    license='MIT License',
    python_requires='>=3.9',
    packages=['redphot'],
    include_package_data=True,
    package_data={'redphot': ['ref_data/*']},
    install_requires=[
        'numpy>=1.23',
        'matplotlib>=3.6',
        'astropy>=5.2',
        'astroquery>=0.4.6',
        'photutils>=1.9',
        'scipy>=1.9',
    ],
    extras_require={
        'cosmic_rays': ['astroscrappy>=1.1'],
        'test': ['pytest>=7'],
    },
)
