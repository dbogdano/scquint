from setuptools import setup


install_requires = [
    "anndata",
    "scanpy",
    "networkx",
    "torch",
    "pyro-ppl",
    "scikit-learn",
    "snakemake",
    "statsmodels",
    "joblib",
    "numpy",
    "pandas",
    "scipy",
    "tqdm",
    "scvi-tools>=1.4.2",
    "zarr<3",
]


setup(
    name='scquint',
    version='0.3.3',
    description='scQuint',
    url='http://github.com/songlab-cal/scquint',
    author='Gonzalo Benegas',
    author_email='gbenegas@berkeley.edu',
    license='MIT',
    packages=['scquint', 'scquint.dimensionality_reduction'],
    python_requires='>=3.12',
    zip_safe=False,
    install_requires=install_requires,
    extras_require = {
        'vae':  ["scvi-tools>=1.4.2"],
    }
)
