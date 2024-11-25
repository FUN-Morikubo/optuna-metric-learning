from setuptools import setup, find_namespace_packages

requirements=[
    "numpy",
    "optuna",
    "pytorch-metric-learning",
    "radam @ git+https://github.com/LiyuanLucasLiu/RAdam.git",
    "RandAugment @ git+https://github.com/ildoonet/pytorch-randaugment.git",
    # "scikit-learn",
    "torch",
    "torchvision",
    "record-keeper",
    "tensorboard",
    "faiss-cpu",
    "timm @ git+https://github.com/rwightman/pytorch-image-models.git"
]

setup(
    name="optuna-metric-learning",
    version="0.0",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements
)