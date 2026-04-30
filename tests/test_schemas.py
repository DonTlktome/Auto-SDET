import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Optional
from pydantic import ValidationError

from auto_sdet.models.schemas import (
    FileContent,
    ExecutionResult,
    AgentState
)


class TestFileContent:
    """Test suite for FileContent model."""
    
    @pytest.mark.parametrize(
        "path, content, mime_type, expected_mime_type",
        [
            ("/home/user/test.py", "print('hello')", "text/plain", "text/plain"),
            ("/home/user/test.py", "print('hello')", "text/x-python", "text/x-python"),
            ("/home/user/test.py", "print('hello')", None, "text/plain"),
            ("relative/path.py", "content", "text/plain", "text/plain"),
        ]
    )
    def test_creation_happy_path(self, path, content, mime_type, expected_mime_type):
        """Test FileContent creation with various valid inputs."""
        if mime_type is None:
            file_content = FileContent(path=path, content=content)
        else:
            file_content = FileContent(path=path, content=content, mime_type=mime_type)
        
        assert file_content.path == path
        assert file_content.content == content
        assert file_content.mime_type == expected_mime_type
    
    @pytest.mark.parametrize(
        "path, content, mime_type, expected_error",
        [
            (None, "content", "text/plain", "Input should be a valid string"),
            ("/path.py", None, "text/plain", "Input should be a valid string"),
            # Note: Empty strings are allowed since there's no min_length constraint
            # ("", "content", "text/plain", "path\n  String should have at least 1 character"),
            # ("/path.py", "", "text/plain", "content\n  String should have at least 1 character"),
            # ("/path.py", "content", "", "mime_type\n  String should have at least 1 character"),
        ]
    )
    def test_creation_edge_cases(self, path, content, mime_type, expected_error):
        """Test FileContent validation with edge cases and invalid inputs."""
        with pytest.raises(ValidationError) as exc_info:
            FileContent(path=path, content=content, mime_type=mime_type)
        
        assert expected_error in str(exc_info.value)
    
    def test_default_mime_type(self):
        """Test that mime_type defaults to 'text/plain' when not provided."""
        file_content = FileContent(path="/test.py", content="test")
        assert file_content.mime_type == "text/plain"
    
    def test_model_dump(self):
        """Test serialization of FileContent model."""
        file_content = FileContent(
            path="/home/user/test.py",
            content="print('hello')",
            mime_type="text/x-python"
        )
        
        dumped = file_content.model_dump()
        assert dumped == {
            "path": "/home/user/test.py",
            "content": "print('hello')",
            "mime_type": "text/x-python"
        }


class TestExecutionResult:
    """Test suite for ExecutionResult model."""
    
    @pytest.mark.parametrize(
        "stdout, stderr, exit_code, duration_ms, sandbox_id, coverage_pct",
        [
            ("output", "error", 0, 100, "sandbox-123", 85),
            ("", "", 1, 0, "", None),
            ("multi\nline\noutput", "traceback\nerror", 2, 500, "sandbox-456", 0),
            ("output", "error", -1, 1000, "sandbox-789", 100),
        ]
    )
    def test_creation_happy_path(self, stdout, stderr, exit_code, duration_ms, sandbox_id, coverage_pct):
        """Test ExecutionResult creation with various valid inputs."""
        result = ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_ms=duration_ms,
            sandbox_id=sandbox_id,
            coverage_pct=coverage_pct
        )
        
        assert result.stdout == stdout
        assert result.stderr == stderr
        assert result.exit_code == exit_code
        assert result.duration_ms == duration_ms
        assert result.sandbox_id == sandbox_id
        assert result.coverage_pct == coverage_pct
    
    @pytest.mark.parametrize(
        "stdout, stderr, exit_code, duration_ms, sandbox_id, coverage_pct, expected_error",
        [
            (None, "error", 0, 100, "sandbox", 85, "Input should be a valid string"),
            ("output", None, 0, 100, "sandbox", 85, "Input should be a valid string"),
            ("output", "error", None, 100, "sandbox", 85, "Input should be a valid integer"),
            ("output", "error", "not_int", 100, "sandbox", 85, "Input should be a valid integer"),
            ("output", "error", 0, "not_int", "sandbox", 85, "Input should be a valid integer"),
            # Note: Negative duration_ms is allowed since there's no ge constraint
            # ("output", "error", 0, -100, "sandbox", 85, "duration_ms\n  Input should be greater than or equal to 0"),
            ("output", "error", 0, 100, None, 85, "Input should be a valid string"),
            # Note: Negative coverage_pct is allowed since there's no ge constraint
            # ("output", "error", 0, 100, "sandbox", -1, "coverage_pct\n  Input should be greater than or equal to 0"),
            # Note: coverage_pct > 100 is allowed since there's no le constraint
            # ("output", "error", 0, 100, "sandbox", 101, "coverage_pct\n  Input should be less than or equal to 100"),
        ]
    )
    def test_creation_edge_cases(self, stdout, stderr, exit_code, duration_ms, sandbox_id, coverage_pct, expected_error):
        """Test ExecutionResult validation with edge cases and invalid inputs."""
        with pytest.raises(ValidationError) as exc_info:
            ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                duration_ms=duration_ms,
                sandbox_id=sandbox_id,
                coverage_pct=coverage_pct
            )
        
        assert expected_error in str(exc_info.value)
    
    def test_default_values(self):
        """Test that ExecutionResult fields have correct default values."""
        result = ExecutionResult(exit_code=0)
        
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.duration_ms == 0
        assert result.sandbox_id == ""
        assert result.coverage_pct is None
    
    def test_model_dump_with_defaults(self):
        """Test serialization of ExecutionResult with default values."""
        result = ExecutionResult(exit_code=1)
        
        dumped = result.model_dump()
        assert dumped == {
            "stdout": "",
            "stderr": "",
            "exit_code": 1,
            "duration_ms": 0,
            "sandbox_id": "",
            "coverage_pct": None
        }
    
    def test_coverage_pct_optional(self):
        """Test that coverage_pct can be omitted entirely."""
        result = ExecutionResult(exit_code=0)
        assert result.coverage_pct is None


class TestAgentState:
    """Test suite for AgentState TypedDict."""
    
    def test_creation_happy_path(self):
        """Test AgentState creation with all fields populated."""
        state: AgentState = {
            "source_code": "def test(): pass",
            "source_path": "/home/user/test.py",
            "context_files": {"file1.py": "content1", "file2.py": "content2"},
            "test_code": "def test_function(): assert True",
            "execution_result": ExecutionResult(exit_code=0, stdout="All tests passed"),
            "retry_count": 2,
            "max_retries": 5,
            "error_history": ["Error 1", "Error 2"],
            "status": "executing"
        }
        
        assert state["source_code"] == "def test(): pass"
        assert state["source_path"] == "/home/user/test.py"
        assert len(state["context_files"]) == 2
        assert state["test_code"] == "def test_function(): assert True"
        assert state["execution_result"].exit_code == 0
        assert state["retry_count"] == 2
        assert state["max_retries"] == 5
        assert len(state["error_history"]) == 2
        assert state["status"] == "executing"
    
    def test_creation_partial_fields(self):
        """Test AgentState creation with only required fields (TypedDict with total=False)."""
        state: AgentState = {
            "source_code": "def test(): pass",
            "source_path": "/home/user/test.py",
            "status": "generating"
        }
        
        assert state["source_code"] == "def test(): pass"
        assert state["source_path"] == "/home/user/test.py"
        assert state["status"] == "generating"
        assert "test_code" not in state
        assert "execution_result" not in state
        assert "retry_count" not in state
    
    def test_empty_state(self):
        """Test that AgentState can be completely empty due to total=False."""
        state: AgentState = {}
        
        assert len(state) == 0
    
    @pytest.mark.parametrize(
        "status_value",
        [
            "generating",
            "executing",
            "reflecting",
            "done",
            "failed",
        ]
    )
    def test_status_literal_values(self, status_value):
        """Test that status field accepts all literal values."""
        state: AgentState = {"status": status_value}
        assert state["status"] == status_value
    
    def test_invalid_status_value(self):
        """Test that invalid status values are caught by type checking (runtime)."""
        # Note: This test demonstrates that invalid values would be caught by type checkers
        # At runtime, Python won't enforce Literal types without additional validation
        state: AgentState = {"status": "invalid_status"}  # type: ignore
        
        # The assignment above would be caught by mypy/pyright but not at runtime
        # This is expected behavior for TypedDict with Literal
        assert state["status"] == "invalid_status"
    
    def test_context_files_structure(self):
        """Test that context_files maintains proper dict structure."""
        context_files: Dict[str, str] = {
            "module1.py": "import os",
            "module2.py": "def func(): return 42",
            "__init__.py": ""
        }
        
        state: AgentState = {
            "context_files": context_files,
            "source_code": "main content",
            "source_path": "/main.py"
        }
        
        assert isinstance(state["context_files"], dict)
        assert len(state["context_files"]) == 3
        assert state["context_files"]["module2.py"] == "def func(): return 42"
    
    def test_error_history_list_operations(self):
        """Test that error_history supports list operations."""
        state: AgentState = {
            "error_history": ["First error", "Second error"],
            "source_code": "test",
            "source_path": "/test.py"
        }
        
        # Test appending to error_history
        state["error_history"].append("Third error")
        assert len(state["error_history"]) == 3
        assert state["error_history"][2] == "Third error"
    
    def test_execution_result_optional(self):
        """Test that execution_result can be None or ExecutionResult instance."""
        # Test with None
        state_with_none: AgentState = {
            "execution_result": None,
            "source_code": "test",
            "source_path": "/test.py"
        }
        assert state_with_none["execution_result"] is None
        
        # Test with ExecutionResult instance
        result = ExecutionResult(exit_code=0, stdout="Success")
        state_with_result: AgentState = {
            "execution_result": result,
            "source_code": "test",
            "source_path": "/test.py"
        }
        assert state_with_result["execution_result"] == result
        assert state_with_result["execution_result"].exit_code == 0
    
    def test_retry_count_default_behavior(self):
        """Test that retry_count can be omitted or set to various values."""
        # Test without retry_count
        state_without: AgentState = {
            "source_code": "test",
            "source_path": "/test.py"
        }
        assert "retry_count" not in state_without
        
        # Test with retry_count = 0
        state_with_zero: AgentState = {
            "retry_count": 0,
            "source_code": "test",
            "source_path": "/test.py"
        }
        assert state_with_zero["retry_count"] == 0
        
        # Test with positive retry_count
        state_with_positive: AgentState = {
            "retry_count": 3,
            "source_code": "test",
            "source_path": "/test.py"
        }
        assert state_with_positive["retry_count"] == 3
    
    def test_nested_model_in_state(self):
        """Test that ExecutionResult model can be nested within AgentState."""
        execution_result = ExecutionResult(
            exit_code=1,
            stdout="Test output",
            stderr="AssertionError",
            duration_ms=150,
            sandbox_id="e2b-123",
            coverage_pct=75
        )
        
        state: AgentState = {
            "source_code": "def broken(): assert False",
            "source_path": "/broken.py",
            "test_code": "def test_broken(): broken()",
            "execution_result": execution_result,
            "retry_count": 1,
            "max_retries": 3,
            "error_history": ["Test failed with AssertionError"],
            "status": "reflecting"
        }
        
        assert state["execution_result"].exit_code == 1
        assert state["execution_result"].stdout == "Test output"
        assert state["execution_result"].coverage_pct == 75
        assert state["retry_count"] == 1
        assert state["status"] == "reflecting"