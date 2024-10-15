"""Handle multiple platforms for the dependencies with rye.

Use the `cereals` key in pyproject toml to collect configurations, like this

[cereals.cpu]
dependencies = [ "torch==2.3.1+cpu",]
extra_index_urls = ["https://download.pytorch.org/whl/cpu"]

[cereals.cu121]
dependencies = ["torch==2.3.1+cu121"]
extra_index_urls = ["https://download.pytorch.org/whl/cu121"]

the different sets of additional dependencies can be declared like this

[project.optional-dependencies]
dev= [
    "jupyterlab>=4.0",
]
ci = [
    "pytest-cov>=5.0.0",
]

it's assumed that there is at least the dev section present.

This script can be used to generate lock files for all platforms
and then sync the venv to the chosen platform.
For doing the lock-file generation it needs the tomlkit package.
This is imported inside the functions, so that some functionality
can be used without tomlkit installed. (important for the CI pipeline)
"""

import os
import shutil
import subprocess
from itertools import chain, repeat
from logging import getLogger
from pathlib import Path

from tomlkit import TOMLDocument

# path to rye callable
UV = os.getenv("UV_PATH", shutil.which("uv"))

DEFAULT_PLATFORM = os.getenv("PLATFORM", "cu121")

logger = getLogger(__name__)


def load_pyproject_toml() -> TOMLDocument:
    """Load pyproject.toml file."""
    import tomlkit

    with open("pyproject.toml") as f:
        return tomlkit.load(f)


def write_pyproject_toml(data: TOMLDocument):
    """Write data to pyproject.toml file."""
    import tomlkit

    with open("pyproject.toml", "w") as f:
        tomlkit.dump(data, f)


def get_platforms(data: TOMLDocument) -> list:
    """Get all platforms from cereals section of pyproject.toml."""
    return list(data.get("cereals", {}).keys())


def get_platform_info(platform: str, data: TOMLDocument) -> tuple[list, list]:
    """Get extra_index_urls and dependencies for a specific platform."""
    extra_index_urls = data.get("cereals").get(platform).get("extra_index_urls", [])
    dependencies = data.get("cereals").get(platform).get("dependencies", [])
    return extra_index_urls, dependencies


def get_extras(data: TOMLDocument) -> list:
    """Get the keys for the optional dependencies in pyproject.toml."""
    return data["project"].get("optional-dependencies", {}).keys()


def get_project_dependencies(data: TOMLDocument) -> list:
    """Get sources and dependencies from the project section of pyproject.toml."""
    project_dependencies = data["project"].get("dependencies", [])
    return project_dependencies


def set_project_dependencies(data: TOMLDocument, project_dependencies: list) -> TOMLDocument:
    """Overwrite project.dependencies sections in pyproject.toml."""
    data["project"]["dependencies"] = project_dependencies
    return data


def set_platform_in_toml(platform: str, data: TOMLDocument) -> TOMLDocument:
    """Set sources and dependencies from specific platform.

    This is reading from the cereals section of pyproject.toml and adding it to the
    existing project sources and dependencies.
    """
    _, dependencies = get_platform_info(platform, data)
    project_dependencies = get_project_dependencies(data)
    project_dependencies.extend(dependencies)
    project_dependencies = project_dependencies.multiline(True)
    return set_project_dependencies(data, project_dependencies)


def unset_platform_in_toml(platform: str, data: TOMLDocument) -> TOMLDocument:
    """Remove sources and dependencies for the given platform from the project settings.

    This is reading from the cereals section of pyproject.toml and removing these from
    the project sources and dependencies.
    """
    _, platform_dependencies = get_platform_info(platform, data)
    project_dependencies = get_project_dependencies(data)
    project_dependencies = list(
        filter(lambda x: x not in platform_dependencies, project_dependencies)
    )
    return set_project_dependencies(data, project_dependencies)


def generate_lock_files_uv(extra_index_urls: list, outpath: str, additional_args: list = []):
    """Generate lock files using rye."""
    urls = chain(*zip(repeat("--extra-index-url"), extra_index_urls))
    subprocess.call(
        [
            UV,
            "pip",
            "compile",
            "pyproject.toml",
            "--emit-index-url",
            "--index-strategy",
            "unsafe-first-match",
            *urls,
            *additional_args,
            "-o",
            outpath,
        ]
    )


def generate_lock_files(platform: str, data: TOMLDocument, arguments: list = None):
    """Move lock files to a subdirectory that is specific to the platform."""
    if arguments is None:
        arguments = []
    target = Path("requirements") / platform
    target.mkdir(exist_ok=True, parents=True)
    extra_index_urls, _ = get_platform_info(platform, data)
    generate_lock_files_uv(
        extra_index_urls=extra_index_urls,
        outpath=target / "requirements.lock",
        additional_args=arguments,
    )

    all_extras = get_extras(data)
    for extra in all_extras:
        generate_lock_files_uv(
            extra_index_urls=extra_index_urls,
            outpath=target / f"requirements-{extra}.lock",
            additional_args=["--extra", extra] + arguments,
        )
    # and all extras
    additional_args = list(chain(*zip(repeat("--extra"), all_extras)))
    generate_lock_files_uv(
        extra_index_urls=extra_index_urls,
        outpath=target / "requirements-all.lock",
        additional_args=additional_args + arguments,
    )


def call_uv_sync(platform, lockfile: str = "requirements.lock"):
    """Call rye sync.

    Only sync the venv to the versions given in the lock files,
    do not update the lock files.
    """
    lockfile = str(Path("requirements") / platform / lockfile)
    subprocess.call([UV, "pip", "sync", lockfile])


def set_platform(platform: str):
    """Set the platform in pyproject.toml.

    Use existing lockfiles for this platform.
    """
    data = load_pyproject_toml()
    platforms = get_platforms(data)
    logger.info("Use the platforms:" + ", ".join(platforms))
    for p in platforms:
        unset_platform_in_toml(p, data)
    data = set_platform_in_toml(platform, data)
    write_pyproject_toml(data)
    logger.info(f"Platform {platform} set.")


def lock(arguments: list):
    """Generate lock files for all platforms.

    use arguments as additional arguments to the uv-call
    """
    data = load_pyproject_toml()
    platforms = get_platforms(data)
    logger.debug("Using the platforms:" + ", ".join(platforms))
    for p in platforms:
        unset_platform_in_toml(p, data)
    for p in platforms:
        set_platform_in_toml(p, data)
        write_pyproject_toml(data)
        generate_lock_files(p, data, arguments)
        unset_platform_in_toml(p, data)
        write_pyproject_toml(data)

    # back to default
    set_platform_in_toml(DEFAULT_PLATFORM, data)
    write_pyproject_toml(data)


def sync():
    """Sync the venv to the versions given in the lock files."""
    call_uv_sync(DEFAULT_PLATFORM)
    logger.info("uv sync completed.")


def sync_dev():
    """Sync the venv to the versions given in the lock files."""
    call_uv_sync(DEFAULT_PLATFORM, lockfile="requirements-dev.lock")
    logger.info("uv sync (dev environment) completed.")


if __name__ == "__main__":
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else None
    match command:
        case "lock":
            arguments = sys.argv[2:] if len(sys.argv) > 2 else []
            lock(arguments)
        case "sync":
            sync()
        case "set_platform":
            if len(sys.argv) < 2:
                raise ValueError("Platform not provided.")
            platform = sys.argv[2]
            set_platform(platform)
