"""Checks on how perception sources are assembled into the sandbox bootstrap.

The sandbox cannot import project code, so each perception module's source is pasted into
the bootstrap with its intra-package imports stripped. That assembly step is easy to break
from a distance -- by adding a module, or by writing an import in a shape the stripper does
not recognise -- so it is pinned here rather than only exercised indirectly.
"""
import importlib.util
import pkgutil
import sys

import pytest

pytest.importorskip(
    "inference.agent.python_tool_sandbox",
    reason="requires the project environment (analyzer dependencies installed)",
)

import inference.perception  # noqa: E402
from inference.agent.python_tool_sandbox import (  # noqa: E402
    _PERCEPTION_MODULES,
    _SANDBOX_BOOTSTRAP,
    _sandbox_source,
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[name]
    return module


def test_bootstrap_compiles():
    compile(_SANDBOX_BOOTSTRAP, "<sandbox_bootstrap>", "exec")


def test_no_project_import_survives_into_the_bootstrap():
    for line in _SANDBOX_BOOTSTRAP.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("from inference"), line
        assert not stripped.startswith("import inference"), line


def test_every_perception_module_is_spliced():
    available = {
        name
        for _finder, name, _is_package in pkgutil.iter_modules(inference.perception.__path__)
    }
    spliced = {module.__name__.rsplit(".", 1)[-1] for module in _PERCEPTION_MODULES}

    assert available == spliced, "a perception module is missing from _PERCEPTION_MODULES"


def test_public_helpers_all_reach_the_bootstrap():
    for name in inference.perception.__all__:
        assert f"def {name}(" in _SANDBOX_BOOTSTRAP, name


def test_single_line_project_imports_are_stripped(tmp_path):
    path = tmp_path / "single.py"
    path.write_text(
        "import math\n"
        "from inference.perception.grid import _dims\n"
        "import inference.perception.grid\n"
        "\n"
        "def area(grid):\n"
        "    return math.prod(_dims(grid))\n"
    )

    source = _sandbox_source(_load(path, "single"))

    assert "import math" in source
    assert "inference" not in source
    assert "def area(grid):" in source


def test_parenthesized_project_imports_are_stripped_in_full(tmp_path):
    path = tmp_path / "wrapped.py"
    path.write_text(
        "from inference.perception.grid import (\n"
        "    _cell_char,\n"
        "    _dims,\n"
        ")\n"
        "import json\n"
        "\n"
        "def encode(grid):\n"
        "    return json.dumps(_dims(grid))\n"
    )

    source = _sandbox_source(_load(path, "wrapped"))

    # A dangling name list would be a syntax error once spliced.
    compile(source, "<wrapped>", "exec")
    assert "_cell_char" not in source
    assert "import json" in source


def test_stdlib_imports_are_left_alone(tmp_path):
    path = tmp_path / "stdlib_only.py"
    path.write_text("import hashlib\nimport json\n\n\ndef digest(value):\n    return value\n")

    source = _sandbox_source(_load(path, "stdlib_only"))

    assert "import hashlib" in source
    assert "import json" in source
