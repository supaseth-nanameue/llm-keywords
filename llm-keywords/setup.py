from setuptools import find_packages, setup

setup(
    name="llm_keywords",
    packages=find_packages(exclude=["llm_keywords_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
