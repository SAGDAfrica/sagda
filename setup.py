from setuptools import setup, find_packages

setup(
    name='sagda',
    version='0.1.0',
    author='PySAGDA',
    author_email='abdelghani.belgaid@um6p.ma',
    description='Generating and Augmenting Agricultural Synthetic Data in Africa',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/PySAGDA/sagda/',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'tensorflow',
        'requests',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
