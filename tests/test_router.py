import pytest
from unittest.mock import patch, MagicMock

from auto_sdet.graph.router import route_after_executor
from auto_sdet.models.schemas import AgentState

# Import the router module to patch its console object
import auto_sdet.graph.router as router_module


class TestRouteAfterExecutor:
    """Test suite for route_after_executor function."""

    @pytest.mark.parametrize(
        "execution_result_exit_code, expected_route",
        [
            (0, "end_success"),
            (1, "reflect"),
            (2, "reflect"),
            (-1, "reflect"),
        ]
    )
    @patch.object(router_module.console, 'print')
    def test_route_with_execution_result(
        self, mock_console_print, execution_result_exit_code, expected_route
    ):
        """Test routing based on execution result exit code."""
        # Arrange
        mock_execution_result = MagicMock()
        mock_execution_result.exit_code = execution_result_exit_code
        
        state: AgentState = {
            "execution_result": mock_execution_result,
            "retry_count": 0,
            "max_retries": 3
        }
        
        # Act
        result = route_after_executor(state)
        
        # Assert
        assert result == expected_route
        mock_console_print.assert_called_once()

    @pytest.mark.parametrize(
        "retry_count, max_retries, expected_route",
        [
            (0, 3, "reflect"),
            (1, 3, "reflect"),
            (2, 3, "reflect"),
            (2, 5, "reflect"),
        ]
    )
    @patch.object(router_module.console, 'print')
    def test_route_with_retries_available(
        self, mock_console_print, retry_count, max_retries, expected_route
    ):
        """Test routing when tests failed but retries are still available."""
        # Arrange
        mock_execution_result = MagicMock()
        mock_execution_result.exit_code = 1
        
        state: AgentState = {
            "execution_result": mock_execution_result,
            "retry_count": retry_count,
            "max_retries": max_retries
        }
        
        # Act
        result = route_after_executor(state)
        
        # Assert
        assert result == expected_route
        mock_console_print.assert_called_once()

    @pytest.mark.parametrize(
        "retry_count, max_retries",
        [
            (3, 3),
            (5, 5),
            (10, 10),
            (1, 1),
        ]
    )
    @patch.object(router_module.console, 'print')
    def test_route_max_retries_reached(
        self, mock_console_print, retry_count, max_retries
    ):
        """Test routing when max retries have been reached."""
        # Arrange
        mock_execution_result = MagicMock()
        mock_execution_result.exit_code = 1
        
        state: AgentState = {
            "execution_result": mock_execution_result,
            "retry_count": retry_count,
            "max_retries": max_retries
        }
        
        # Act
        result = route_after_executor(state)
        
        # Assert
        assert result == "end_failed"
        mock_console_print.assert_called_once()

    @patch.object(router_module.console, 'print')
    def test_route_with_missing_execution_result(self, mock_console_print):
        """Test routing when execution_result is None (edge case)."""
        # Arrange
        state: AgentState = {
            "execution_result": None,
            "retry_count": 0,
            "max_retries": 3
        }
        
        # Act
        result = route_after_executor(state)
        
        # Assert
        assert result == "reflect"
        mock_console_print.assert_called_once()

    @patch.object(router_module.console, 'print')
    def test_route_with_missing_retry_count(self, mock_console_print):
        """Test routing when retry_count is missing from state (edge case)."""
        # Arrange
        mock_execution_result = MagicMock()
        mock_execution_result.exit_code = 1
        
        state: AgentState = {
            "execution_result": mock_execution_result,
            "max_retries": 3
        }
        # Don't add retry_count to test default value
        
        # Act
        result = route_after_executor(state)
        
        # Assert
        assert result == "reflect"
        mock_console_print.assert_called_once()

    @patch.object(router_module.console, 'print')
    def test_route_with_missing_max_retries(self, mock_console_print):
        """Test routing when max_retries is missing from state (edge case)."""
        # Arrange
        mock_execution_result = MagicMock()
        mock_execution_result.exit_code = 1
        
        state: AgentState = {
            "execution_result": mock_execution_result,
            "retry_count": 5
        }
        # Don't add max_retries to test default value
        
        # Act
        result = route_after_executor(state)
        
        # Assert
        assert result == "end_failed"
        mock_console_print.assert_called_once()

    @patch.object(router_module.console, 'print')
    def test_route_with_empty_state(self, mock_console_print):
        """Test routing with minimal state dictionary (edge case)."""
        # Arrange
        state: AgentState = {}
        
        # Act
        result = route_after_executor(state)
        
        # Assert
        assert result == "reflect"
        mock_console_print.assert_called_once()

    @pytest.mark.parametrize(
        "retry_count, max_retries, expected_route",
        [
            (0, 0, "end_failed"),
            (1, 0, "end_failed"),
            (-1, 3, "reflect"),
        ]
    )
    @patch.object(router_module.console, 'print')
    def test_route_with_edge_case_retry_values(
        self, mock_console_print, retry_count, max_retries, expected_route
    ):
        """Test routing with edge case retry count and max retries values."""
        # Arrange
        mock_execution_result = MagicMock()
        mock_execution_result.exit_code = 1
        
        state: AgentState = {
            "execution_result": mock_execution_result,
            "retry_count": retry_count,
            "max_retries": max_retries
        }
        
        # Act
        result = route_after_executor(state)
        
        # Assert
        assert result == expected_route
        mock_console_print.assert_called_once()

    @patch.object(router_module.console, 'print')
    def test_route_success_with_additional_state_fields(self, mock_console_print):
        """Test successful routing with additional fields in state (edge case)."""
        # Arrange
        mock_execution_result = MagicMock()
        mock_execution_result.exit_code = 0
        
        state: AgentState = {
            "execution_result": mock_execution_result,
            "retry_count": 2,
            "max_retries": 3,
            "source_path": "/some/path",
            "test_code": "def test(): pass",
            "error_history": [],
            "status": "executing"
        }
        
        # Act
        result = route_after_executor(state)

        # Assert
        assert result == "end_success"
        mock_console_print.assert_called_once()
