from setuptools import setup, find_packages


setup(name='pyFernando',
      version='0.1',
      packages=find_packages(),
      install_requires=['tqdm','numpy','seaborn','polars'],
      author='Fernando Fernandes',
      author_email='fernando.allysson@usp.br',
      description='this packages is useful for plots',
      url='https://github.com/fernandoACF28/pyFernando',
      classifiers=['Programing Language :: Python :: 3',
                   'License :: OSI Approved :: MIT license',],)