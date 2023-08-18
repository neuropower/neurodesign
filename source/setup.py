from setuptools import setup

import os.path as op

with open('requirements.txt') as rf:
    requirements = rf.readlines()

setup(name='neurodesign',
      version='0.2.02',
      description='Package for design optimisation for fMRI experiments',
      author='Joke Durnez',
      author_email='joke.durnez@gmail.com',
      license='MIT',
      packages=['neurodesign'],
      install_required=requirements,
      package_dir={'neurodesign':'src'},
      package_data={'neurodesign':['media/NeuroDes.png']},
      zip_safe=False)
