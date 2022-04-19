"""Nox configuration."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import nox

try:
    from nox_poetry import Session, session
except ImportError:
    message = f"""\
    Nox failed to import the 'nox-poetry' package.
    Please install it using the following command:
    {sys.executable} -m pip install nox-poetry"""
    raise SystemExit(dedent(message)) from None


package = "sensopy"
python_versions = ["3.10", "3.9", "3.8"]
locations = "src", "tests", "noxfile.py"
nox.options.sessions = ("tests",)


@session(python=python_versions)
def tests(session: Session) -> None:
    """Execute pytest tests."""
    session.install(".")
    session.install("coverage[toml]", "pytest")
    try:
        session.run("coverage", "run", "--parallel", "-m", "pytest", *session.posargs)
    finally:
        if session.interactive:
            session.notify("coverage", posargs=[])


@session(python=python_versions[0])
def coverage(session: Session) -> None:
    """Upload coverage data."""
    args = session.posargs or ["report"]

    session.install("coverage[toml]")

    if not session.posargs and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")

    session.run("coverage", *args)
