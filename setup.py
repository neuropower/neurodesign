from setuptools import setup, find_packages

packages = find_packages()

setup(name='neurodesign',
      version='0.2.02',
      description='Package for design optimisation for fMRI experiments',
      author='Joke Durnez',
      author_email='joke.durnez@gmail.com',
      license='MIT',
      packages=pakcages,
      install_required=[
          'numpy>1.0.0',
          'scipy>1.0.0',
          'sklearn>0.15.0',
          'pandas>0.15.0',
          'progressbar>2.0',
          'math',
          'reportlab',
          'matplotlib',
          'seaborn'
          ],
      zip_safe=False)
