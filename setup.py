from setuptools import setup

setup(name='redphot',
      version='0.1',
      description='Photometry Pipeline',
      url='https://github.com/gmzsebastian/redphot',
      author=['Sebastian Gomez'],
      author_email=['sgomez@cfa.harvard.edu',],
      license='GNU GPL 3.0',
      packages=['redphot'],
      install_requires=[
          'numpy',
          'matplotlib',
          'astropy',
          'photutils',
          'scipy',
          'mastcasjobs'
      ],
      test_suite='nose.collector',
      zip_safe = False)