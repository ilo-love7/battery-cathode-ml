"""Setup configuration for battery-cathode-ml package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="battery-cathode-ml",
    version="1.0.0",
    author="Battery Cathode ML Contributors",
    author_email="17695324ilo@gmail\.com",  # Replace with your email
    description="Machine learning toolkit for battery cathode material property prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ilo-love7/battery-cathode-ml",  # Replace with your repo URL
    packages=find_packages(where="src"),
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
        "vision": [
            "pymatgen>=2023.0.0",
            "mp-api>=0.41.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "battery-predict=battery_cathode_ml.predictor:main",
            "battery-train=battery_cathode_ml.train_voltage:main",
        ],
    },
    include_package_data=True,
    keywords=[
        "machine learning",
        "battery",
        "cathode",
        "materials science",
        "chemistry",
        "prediction",
        "energy storage",
    ],
)
