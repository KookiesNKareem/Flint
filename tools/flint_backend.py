"""PEP 517 bridge from the research checkout to Flint's standalone distribution."""
from contextlib import contextmanager
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from setuptools import build_meta

from build_flint_package import stage


@contextmanager
def staged_sources():
    previous = Path.cwd()
    with TemporaryDirectory(prefix="flint-package-") as scratch:
        source = stage(Path(scratch) / "source")
        os.chdir(source)
        try:
            yield
        finally:
            os.chdir(previous)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    wheel_directory = str(Path(wheel_directory).resolve())
    metadata_directory = str(Path(metadata_directory).resolve()) if metadata_directory else None
    with staged_sources():
        return build_meta.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    sdist_directory = str(Path(sdist_directory).resolve())
    with staged_sources():
        return build_meta.build_sdist(sdist_directory, config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    metadata_directory = str(Path(metadata_directory).resolve())
    with staged_sources():
        return build_meta.prepare_metadata_for_build_wheel(metadata_directory, config_settings)
