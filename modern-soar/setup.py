
#!/usr/bin/env python3
import os
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="modern-soar",
    version="1.0.0",
    author="AI Research Team",
    author_email="research@example.com",
    description="Modern Soar Cognitive Architecture with AI Integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/modern-soar",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.1.0',
            'pytest-asyncio>=0.19.0',
            'black>=22.6.0',
            'isort>=5.10.0',
            'mypy>=0.971.0',
        ],
        'docs': [
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.0.0',
            'myst-parser>=0.18.0',
        ],
        'viz': [
            'graphviz>=0.20.0',
            'pydot>=1.4.0',
            'plotly>=5.10.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'soar-demo=modern_soar.cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
