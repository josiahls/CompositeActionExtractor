from setuptools import setup, find_packages
import sys, os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'composite_action_extractor'))

VERSION = 0.1

setup_py_dir = os.path.dirname(os.path.realpath(__file__))

setup(name='composite_action_extractor',
      version=VERSION,
      description='Extracts action segments from state and action chains.',
      url='https://github.com/josiahls/Composite-Action-Extractor',
      author='Josiah Laivins',
      author_email='jlaivins@uncc.edu',
      license='',
      packages=[package for package in find_packages()
                if package.startswith('composite_action_extractor')],
      zip_safe=False,
      dependency_links=['https://github.com/MattChanTK/gym-maze/tarball/master#egg=gym-maz-1.0'],
      install_requires=['numpy', 'tqdm', 'pillow', 'pandas'],
      )
