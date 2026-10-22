"""Setup configuration for battery-cathode-ml.

本项目采用「扁平 src 布局」：所有模块都在 ``src/`` 下且互相以顶层模块名
导入（如 ``from features import featurize_one``）。因此这里用 ``py_modules``
+ ``package_dir`` 而不是 ``find_packages``——后者在 src/ 没有子包时返回空列表，
会让 ``pip install .`` 装出一个空包（1.0.0 就是这个问题）。

安装后可直接::

    from predictor import predict_material_properties
    # 或命令行
    battery-predict --formulas "LiFePO4,LiCoO2"
"""

from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).resolve().parent
long_description = (ROOT / "README.md").read_text(encoding="utf-8")
requirements = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]

# src/ 下的顶层模块（与 src/ 目录一一对应）
PY_MODULES = [
    "cgcnn_data",
    "cgcnn_model",
    "conformal",
    "console",
    "download_data",
    "element_data",
    "export_cgcnn",
    "features",
    "fetch_mp_battery",
    "fetch_mp_structures",
    "fetch_mp_data",
    "formula",
    "generate_voltage_data",
    "models",
    "predictor",
    "predict_cgcnn",
    "predict_voltage_cgcnn",
    "prepare_data",
    "prepare_mp_data",
    "prepare_mp_summary",
    "prepare_struct_features",
    "prepare_voltage_struct",
    "screening",
    "screening_struct",
    "generate_prototype_structures",
    "prototype_sensitivity",
    "template_substitution",
    "drx_ensemble",
    "structure_features",
    "train_capacity",
    "train_cgcnn",
    "train_cgcnn_mp",
    "train_cgcnn_voltage",
    "train_ef",
    "train_ef_struct",
    "train_gap",
    "train_voltage",
    "train_voltage_struct",
    "validate_known_cathodes",
    "validate_mp_external",
    "voltage",
    "voltage_from_cgcnn",
]

setup(
    name="battery-cathode-ml",
    version="1.13.0",
    author="Battery Cathode ML Contributors",
    author_email="17695324ilo@gmail.com",
    description="Machine learning toolkit for battery cathode material property prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ilo-love7/battery-cathode-ml",
    project_urls={
        "Bug Tracker": "https://github.com/ilo-love7/battery-cathode-ml/issues",
        "Documentation": "https://github.com/ilo-love7/battery-cathode-ml#readme",
        "Source": "https://github.com/ilo-love7/battery-cathode-ml",
    },
    py_modules=PY_MODULES,
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Materials Science",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
            "bandit>=1.7.0",
        ],
        "mp": [
            "pymatgen>=2023.0.0",
            "mp-api>=0.41.0",
        ],
        "gpu": [
            "torch>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "battery-predict=predictor:main",
            "battery-train-voltage=train_voltage:main",
        ],
    },
    include_package_data=True,
    keywords=[
        "machine learning", "battery", "cathode", "materials science",
        "chemistry", "prediction", "energy storage", "materials informatics",
    ],
)
