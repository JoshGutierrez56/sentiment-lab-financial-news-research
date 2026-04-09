from setuptools import setup, find_packages

setup(
    name="news-sentiment-trader",
    version="0.1.0",
    description="Algorithmic trading system using FactSet news + DeepSeek sentiment",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "requests>=2.31",
        "python-dotenv>=1.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4", "pytest-cov>=4.0"],
    },
)
