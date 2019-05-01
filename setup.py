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
      license='MIT',
      install_requires=[
          'pytest>4.1.0',
          'pylint'
      ],
      python_requires='>=3.6.1',
      packages=find_packages(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python :: 3.6',
          'Topic :: Text Processing',
          'Topic :: Text Processing :: Linguistic',
      ])
