import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='FLMR',
    version='0.1.0',
    author='Weizhe Lin',
    author_email='wl356@cam.ac.uk',
    description="Fine-grained Late-interaction Multi-modal Retrieval",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/LinWeizheDragon/FLMR/',
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
)
