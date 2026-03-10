from setuptools import setup, find_packages


def read_requirements():
    with open("requirements.txt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="semen",
    version="0.1.0",
    description="SEgmentation for MEthylation Noise reduction",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=read_requirements(),
)
