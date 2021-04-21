from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()


setup(
    name="cida-unofficial",
    version="0.0.0",
    description="An unofficial refactor of Continuously Indexed Domain Adaptation.",
    url="https://github.com/sshh12/cida",
    license="MIT",
    packages=["cida"],
    include_package_data=True,
    install_requires=required,
)