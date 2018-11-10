from setuptools import setup, find_packages

packages = find_packages()

with open('requirements.txt') as rf:
    requirements = rf.readlines()

setup(name='neurodesign',
      version='0.2.2',
      description='Package for design optimisation for fMRI experiments',
      author='Joke Durnez',
      author_email='joke.durnez@gmail.com',
      license='MIT',
      packages=packages,
      install_requires=requirements,
      zip_safe=False)
