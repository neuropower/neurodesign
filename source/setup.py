from setuptools import setup

setup(name='neurodesign',
      version='0.1.08',
      description='Package for design optimisation for fMRI experiments',
      author='Joke Durnez',
      author_email='joke.durnez@gmail.com',
      license='MIT',
      packages=['neurodesign'],
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
      package_dir={'neurodesign':'src'},
      package_data={'neurodesign':['media/NeuroDes.png']},
      zip_safe=False)
