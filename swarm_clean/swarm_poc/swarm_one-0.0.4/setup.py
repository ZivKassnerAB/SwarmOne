from setuptools import setup, find_packages
import os


VERSION = '0.0.4'
DESCRIPTION = ' '
_CURR_DIRECTORY = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(_CURR_DIRECTORY, "requirements.txt")) as f:
   setup(
      name="swarm_one",
      version=VERSION,
      author="Kim Boren",
      author_email="kimb@r-stealth.com",
      description=DESCRIPTION,
      packages=find_packages(),
      install_requires=f.read().splitlines(),
      # install_requires=["sourcedefender", "pandas", "validators", "numpy", "pytest",
      #                              "fastavro", "progressbar2", "retry", "tensorboardX", "h5py", "importlib_metadata",
      #                              "requests", "setuptools", "Sphinx", "ipython", "prettytable", "ipywidgets"],
      setup_requires=['setuptools'],
      python_requires='>=3.8.0',
      package_data={"swarm_one": ["*.pye"]}
      # include_package_data=True
   )

