from setuptools import setup

with open('requirements.txt', 'r') as file:
    requirements = [line.strip() for line in file if line.strip()]

setup(
    name='NLP Final project',
    description="BERT-like models for SQUAD_v2",
    install_requires=requirements,
)
