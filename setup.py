#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import os

from setuptools import find_packages, find_namespace_packages, setup

SOURCE_DIR = "rl"

if os.path.exists("README.rst"):
    with open("README.rst") as fh:
        readme = fh.read()
else:
    readme = ""
if os.path.exists("HISTORY.md"):
    with open("HISTORY.md") as fh:
        history = fh.read().replace(".. :changelog:", "")
else:
    history = ""


def parse_req(spec: str) -> str:
    """Parse package name==version out of requirements file."""
    if ";" in spec:
        # remove restriction
        spec, _ = [x.strip() for x in spec.split(";", 1)]
    if "#" in spec:
        # remove comment
        spec = spec.strip().split("#")[0]
    if "\\" in spec:
        # remove line breaks
        spec = spec.strip().split("\\")[0]
    if "--hash=" in spec:
        # remove line breaks
        spec = spec.strip().split("--hash=")[0]
    return spec


if os.path.exists("requirements.in"):
    with open("requirements.in") as fh:
        requirements = [parse_req(r) for r in fh.read().replace("\\\n", " ").split("\n") if parse_req(r) != ""]
else:
    requirements = []

# generate extras based on requirements files
extras_require = dict()
for a_extra in ["dev"]:
    req_file = f"requirements.{a_extra}.in"
    if os.path.exists(req_file):
        with open(req_file) as fh:
            extras_require[a_extra] = [r for r in fh.read().split("\n") if ";" not in r]
    else:
        extras_require[a_extra] = []
extras_require["all"] = list(itertools.chain.from_iterable(extras_require.values()))

if os.path.exists("scripts"):
    SCRIPTS = [os.path.join("scripts", a) for a in os.listdir("scripts")]
else:
    SCRIPTS = []

cmdclass = dict()

# Setup package using PIP
if __name__ == "__main__":
    setup(
        name=f"rl",
        version="0.0.1",
        python_requires=">=3.8.0",
        license="Proprietary",
        package_dir=dict([("", "src")]),
        packages=[*find_packages(where="./src/"), *find_namespace_packages(where="./src/", include=["hydra_plugins.*"])],
        scripts=SCRIPTS,
        include_package_data=True,
        install_requires=requirements,
        tests_require=extras_require["dev"],
        extras_require=extras_require,
        cmdclass=cmdclass,
        classifiers=["Private :: Do Not Upload to pypi server"],
    )