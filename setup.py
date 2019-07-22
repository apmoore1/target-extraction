from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(name='target_extraction',
      version='0.0.1',
      description='Target Extraction',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/apmoore1/target-extraction',
      author='Andrew Moore',
      author_email='andrew.p.moore94@gmail.com',
      license='Apache License 2.0',
      install_requires=[
          'spacy>=2.1,<2.2',
          'torch>=0.4.1',
          'stanfordnlp==0.2.0',
          'allennlp>=0.8.3',
          'twokenize>=1.0.0'
      ],
      python_requires='>=3.6.1',
      packages=find_packages(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python :: 3.6'
      ])
