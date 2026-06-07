from pydantic import ValidationError

from agent.models.ast import FindDefinitionInput, ParseModuleOutput
from agent.models.ci import CommandOutput, QualitySummaryOutput
from agent.models.deps import AddDependencyInput, DependencyInfo
from agent.models.test import DiscoverTestsOutput, TestNodeInput


def test_ast_find_definition_requires_name() -> None:
    try:
        FindDefinitionInput(path="module.py", name="")
    except ValidationError:
        return

    raise AssertionError("empty definition name should fail validation")


def test_ast_parse_module_output_shape() -> None:
    output = ParseModuleOutput(path="module.py", node_count=10, syntax_ok=True)

    assert output.success is True
    assert output.node_count == 10


def test_test_node_requires_node_id() -> None:
    try:
        TestNodeInput(node_id="")
    except ValidationError:
        return

    raise AssertionError("empty test node id should fail validation")


def test_discover_tests_output_counts_tests() -> None:
    output = DiscoverTestsOutput(path="tests", tests=["tests/test_app.py::test_ok"], count=1)

    assert output.count == 1


def test_dependency_info_shape() -> None:
    dep = DependencyInfo(name="langgraph", specifier=">=0.2", source="pyproject.toml")

    assert dep.name == "langgraph"


def test_add_dependency_requires_requirement() -> None:
    try:
        AddDependencyInput(requirement="")
    except ValidationError:
        return

    raise AssertionError("empty requirement should fail validation")


def test_ci_command_output_shape() -> None:
    output = CommandOutput(command=["python", "-m", "compileall", "src"], return_code=0, output="")

    assert output.success is True


def test_quality_summary_output_shape() -> None:
    output = QualitySummaryOutput(root=".", checks={"build": True}, summary="ok")

    assert output.checks["build"] is True
