# -*- coding: utf-8 -*-

"""
Tests for kiro/account_manager.py - Unified Account System.

Tests the AccountManager class that manages multiple Kiro accounts with:
- Lazy initialization
- Sticky behavior (prefer successful account)
- Circuit breaker with exponential backoff
- TTL-based model cache refresh
- State persistence
"""

import asyncio
import json
import pytest
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx

from kiro.config import FALLBACK_MODELS
from kiro.account_manager import (
    Account,
    AccountStats,
    ModelAccountList,
    AccountManager,
    _format_duration
)
from kiro.account_errors import ErrorType
from kiro.auth import KiroAuthManager, AuthType
from kiro.cache import ModelInfoCache
from kiro.model_resolver import ModelResolver


class TestAccountDataclass:
    """
    Tests for Account and AccountStats dataclasses.
    """
    
    def test_account_creation_with_defaults(self):
        """
        Test Account creation with default values.
        
        What it does: Verifies Account dataclass initialization
        Purpose: Ensure default values are set correctly
        """
        print("\n=== Test: Account creation with defaults ===")
        
        # Act
        account = Account(id="/test/path.json")
        
        # Assert
        print(f"Account ID: {account.id}")
        print(f"Auth manager: {account.auth_manager}")
        print(f"Failures: {account.failures}")
        print(f"Last failure time: {account.last_failure_time}")
        
        assert account.id == "/test/path.json"
        assert account.auth_manager is None
        assert account.model_cache is None
        assert account.model_resolver is None
        assert account.failures == 0
        assert account.last_failure_time == 0.0
        assert account.models_cached_at == 0.0
        assert isinstance(account.stats, AccountStats)
    
    def test_account_stats_initialization(self):
        """
        Test AccountStats initialization with zeros.
        
        What it does: Verifies AccountStats default values
        Purpose: Ensure statistics start at zero
        """
        print("\n=== Test: AccountStats initialization ===")
        
        # Act
        stats = AccountStats()
        
        # Assert
        print(f"Total requests: {stats.total_requests}")
        print(f"Successful requests: {stats.successful_requests}")
        print(f"Failed requests: {stats.failed_requests}")
        
        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0


class TestAccountManagerLoadCredentials:
    """
    Tests for AccountManager.load_credentials() method.
    """
    
    @pytest.mark.asyncio
    async def test_load_credentials_json_type(self, tmp_path):
        """
        Test loading credentials with type=json.
        
        What it does: Loads single JSON credential file
        Purpose: Verify JSON type credential loading
        """
        print("\n=== Test: load_credentials with type=json ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        credentials = [
            {
                "type": "json",
                "path": str(test_json),
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        print(f"Account IDs: {list(manager._accounts.keys())}")
        
        assert len(manager._accounts) == 1
        assert str(test_json.resolve()) in manager._accounts
    
    @pytest.mark.asyncio
    async def test_load_credentials_sqlite_type(self, tmp_path, temp_sqlite_db):
        """
        Test loading credentials with type=sqlite.
        
        What it does: Loads SQLite database credential
        Purpose: Verify SQLite type credential loading
        """
        print("\n=== Test: load_credentials with type=sqlite ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "sqlite",
                "path": temp_sqlite_db,
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 1
        assert str(Path(temp_sqlite_db).resolve()) in manager._accounts
    
    @pytest.mark.asyncio
    async def test_load_credentials_refresh_token_type(self, tmp_path):
        """
        Test loading credentials with type=refresh_token.
        
        What it does: Loads refresh token credential
        Purpose: Verify refresh_token type credential loading
        """
        print("\n=== Test: load_credentials with type=refresh_token ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "refresh_token",
                "refresh_token": "test_refresh_token_abc123",
                "profile_arn": "arn:aws:codewhisperer:us-east-1:123456789:profile/test",
                "region": "us-east-1",
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        # Create state file to avoid errors
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({"current_account_index": 0, "model_to_accounts": {}, "accounts": {}}))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(state_file)
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        print(f"Account IDs: {list(manager._accounts.keys())}")
        
        assert len(manager._accounts) == 1
        # refresh_token type uses deterministic hash as ID
        account_id = list(manager._accounts.keys())[0]
        assert account_id.startswith("refresh_token_")
    
    @pytest.mark.asyncio
    async def test_load_credentials_folder_scanning(self, tmp_path):
        """
        Test folder scanning for credential files.
        
        What it does: Scans folder and loads all valid credential files
        Purpose: Verify folder scanning functionality
        """
        print("\n=== Test: load_credentials with folder scanning ===")
        
        # Arrange
        folder = tmp_path / "accounts"
        folder.mkdir()
        
        # Create valid files
        file1 = folder / "account1.json"
        file1.write_text(json.dumps({
            "refreshToken": "token1",
            "accessToken": "access1",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        file2 = folder / "account2.json"
        file2.write_text(json.dumps({
            "refreshToken": "token2",
            "accessToken": "access2",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "path": str(folder),
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 2
    
    @pytest.mark.asyncio
    async def test_load_credentials_skip_invalid_files(self, tmp_path):
        """
        Test that invalid files are skipped with WARNING.
        
        What it does: Loads folder with invalid files
        Purpose: Verify invalid files are skipped gracefully
        """
        print("\n=== Test: load_credentials skips invalid files ===")
        
        # Arrange
        folder = tmp_path / "accounts"
        folder.mkdir()
        
        # Valid file
        valid_file = folder / "valid.json"
        valid_file.write_text(json.dumps({
            "refreshToken": "token",
            "accessToken": "access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        # Invalid JSON
        invalid_file = folder / "invalid.json"
        invalid_file.write_text("not a valid json {{{")
        
        # Non-JSON file
        text_file = folder / "readme.txt"
        text_file.write_text("This is not a credential file")
        
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "path": str(folder),
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 1  # Only valid file loaded
    
    @pytest.mark.asyncio
    async def test_load_credentials_skip_disabled(self, tmp_path):
        """
        Test that entries with enabled=false are skipped.
        
        What it does: Loads credentials with disabled entry
        Purpose: Verify enabled flag is respected
        """
        print("\n=== Test: load_credentials skips disabled entries ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "token",
            "accessToken": "access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "path": str(test_json),
                "enabled": False  # Disabled
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0
    
    @pytest.mark.asyncio
    async def test_load_credentials_missing_type(self, tmp_path):
        """
        Test that entries without type are skipped.
        
        What it does: Loads credentials with missing type field
        Purpose: Verify type validation
        """
        print("\n=== Test: load_credentials skips entries without type ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "path": "/some/path.json",
                "enabled": True
                # Missing "type" field
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0
    
    @pytest.mark.asyncio
    async def test_load_credentials_missing_path(self, tmp_path):
        """
        Test that json/sqlite entries without path are skipped.
        
        What it does: Loads credentials with missing path field
        Purpose: Verify path validation for json/sqlite types
        """
        print("\n=== Test: load_credentials skips json/sqlite without path ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "enabled": True
                # Missing "path" field
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0
    
    @pytest.mark.asyncio
    async def test_load_credentials_missing_refresh_token(self, tmp_path):
        """
        Test that refresh_token entries without refresh_token field are skipped.
        
        What it does: Loads credentials with missing refresh_token field
        Purpose: Verify refresh_token validation
        """
        print("\n=== Test: load_credentials skips refresh_token without token ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "refresh_token",
                "profile_arn": "arn:aws:codewhisperer:us-east-1:123456789:profile/test",
                "enabled": True
                # Missing "refresh_token" field
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0
    
    @pytest.mark.asyncio
    async def test_load_credentials_file_not_found(self, tmp_path):
        """
        Test handling of non-existent credentials.json.
        
        What it does: Attempts to load non-existent file
        Purpose: Verify graceful handling of missing file
        """
        print("\n=== Test: load_credentials with missing file ===")
        
        # Arrange
        manager = AccountManager(
            credentials_file=str(tmp_path / "nonexistent.json"),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0


class TestAccountManagerLoadState:
    """
    Tests for AccountManager.load_state() method.
    """
    
    @pytest.mark.asyncio
    async def test_load_state_success(self, tmp_path, sample_state_with_data):
        """
        Test loading existing state.json.
        
        What it does: Loads state from file
        Purpose: Verify state restoration
        """
        print("\n=== Test: load_state success ===")
        
        # Arrange
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(sample_state_with_data))
        
        # Create accounts first
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({"refreshToken": "token"}))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(state_file)
        )
        
        await manager.load_credentials()
        
        # Act
        await manager.load_state()
        
        # Assert
        print(f"Model mappings: {len(manager._model_to_accounts)}")
        print(f"Current account index: {manager._current_account_index}")
        
        assert len(manager._model_to_accounts) > 0
    
    @pytest.mark.asyncio
    async def test_load_state_restore_current_account_index(self, tmp_path):
        """
        Test restoration of global current_account_index.
        
        What it does: Restores sticky index from state
        Purpose: Verify global sticky behavior persistence
        """
        print("\n=== Test: load_state restores current_account_index ===")
        
        # Arrange
        state_data = {
            "current_account_index": 2,
            "model_to_accounts": {},
            "accounts": {}
        }
        
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data))
        
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )
        
        # Act
        await manager.load_state()
        
        # Assert
        print(f"Current account index: {manager._current_account_index}")
        
        assert manager._current_account_index == 2
    
    @pytest.mark.asyncio
    async def test_load_state_restore_model_to_accounts(self, tmp_path):
        """
        Test restoration of model_to_accounts mapping.
        
        What it does: Restores model mappings from state
        Purpose: Verify model-to-account mapping persistence
        """
        print("\n=== Test: load_state restores model_to_accounts ===")
        
        # Arrange
        state_data = {
            "current_account_index": 0,
            "model_to_accounts": {
                "claude-opus-4.5": {
                    "accounts": ["/test/account1.json", "/test/account2.json"]
                }
            },
            "accounts": {}
        }
        
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data))
        
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )
        
        # Act
        await manager.load_state()
        
        # Assert
        print(f"Model mappings: {manager._model_to_accounts}")
        
        assert "claude-opus-4.5" in manager._model_to_accounts
        assert len(manager._model_to_accounts["claude-opus-4.5"].accounts) == 2
    
    @pytest.mark.asyncio
    async def test_load_state_restore_account_runtime_state(self, tmp_path):
        """
        Test restoration of account runtime state (failures, stats, etc).
        
        What it does: Restores account state from file
        Purpose: Verify runtime state persistence
        """
        print("\n=== Test: load_state restores account runtime state ===")
        
        # Arrange
        # Create account first to get correct resolved path
        test_json = tmp_path / "account.json"
        test_json.write_text(json.dumps({"refreshToken": "token"}))
        account_id = str(test_json.resolve())
        
        state_data = {
            "current_account_index": 0,
            "model_to_accounts": {},
            "accounts": {
                account_id: {
                    "failures": 3,
                    "last_failure_time": 1704110400.0,
                    "models_cached_at": 1704106800.0,
                    "stats": {
                        "total_requests": 100,
                        "successful_requests": 97,
                        "failed_requests": 3
                    }
                }
            }
        }
        
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(state_file)
        )
        
        await manager.load_credentials()
        
        # Act
        await manager.load_state()
        
        # Assert
        account = manager._accounts[account_id]
        print(f"Account failures: {account.failures}")
        print(f"Account stats: {account.stats}")
        
        assert account.failures == 3
        assert account.last_failure_time == 1704110400.0
        assert account.models_cached_at == 1704106800.0
        assert account.stats.total_requests == 100
    
    @pytest.mark.asyncio
    async def test_load_state_file_not_found(self, tmp_path):
        """
        Test handling of non-existent state.json (empty state).
        
        What it does: Attempts to load non-existent state file
        Purpose: Verify graceful handling with empty state
        """
        print("\n=== Test: load_state with missing file ===")
        
        # Arrange
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(tmp_path / "nonexistent.json")
        )
        
        # Act
        await manager.load_state()
        
        # Assert
        print(f"Model mappings: {len(manager._model_to_accounts)}")
        print(f"Current account index: {manager._current_account_index}")
        
        assert len(manager._model_to_accounts) == 0
        assert manager._current_account_index == 0
    
    @pytest.mark.asyncio
    async def test_load_state_corrupted_json(self, tmp_path):
        """
        Test handling of corrupted state.json.
        
        What it does: Attempts to load invalid JSON
        Purpose: Verify error handling for corrupted state
        """
        print("\n=== Test: load_state with corrupted JSON ===")
        
        # Arrange
        state_file = tmp_path / "state.json"
        state_file.write_text("not a valid json {{{")
        
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )
        
        # Act
        await manager.load_state()
        
        # Assert - should handle gracefully
        print(f"Model mappings: {len(manager._model_to_accounts)}")
        
        assert len(manager._model_to_accounts) == 0



class TestAccountManagerInitializeAccount:
    """
    Tests for AccountManager._initialize_account() method.
    """
    
    @pytest.mark.asyncio
    async def test_initialize_account_json_success(self, tmp_path, mock_list_models_response):
        """
        Test successful account initialization with type=json.
        
        What it does: Initializes account with JSON credentials
        Purpose: Verify complete initialization flow
        """
        print("\n=== Test: initialize_account with JSON ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z",
            "profileArn": "arn:aws:codewhisperer:us-east-1:123456789:profile/test",
            "region": "us-east-1"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Mock HTTP client for ListAvailableModels
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            # Act
            success = await manager._initialize_account(account_id)
        
        # Assert
        print(f"Initialization success: {success}")
        assert success is True
        assert manager._accounts[account_id].auth_manager is not None
        assert manager._accounts[account_id].model_cache is not None
        assert manager._accounts[account_id].model_resolver is not None
    
    @pytest.mark.asyncio
    async def test_initialize_account_fetch_models_fallback(self, tmp_path):
        """
        Test fallback to FALLBACK_MODELS when API fails.
        
        What it does: Initializes account when ListAvailableModels fails
        Purpose: Verify fallback mechanism
        """
        print("\n=== Test: initialize_account with fallback models ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Mock HTTP client to fail
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_client.request_with_retry = AsyncMock(side_effect=Exception("Network error"))
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            # Act
            success = await manager._initialize_account(account_id)
        
        # Assert
        print(f"Initialization success: {success}")
        assert success is True  # Should succeed with fallback
        assert manager._accounts[account_id].model_cache is not None


class TestAccountManagerGetNextAccount:
    """
    Tests for AccountManager.get_next_account() method.
    """
    
    @pytest.mark.asyncio
    async def test_get_next_account_single_bypass_circuit_breaker(self, tmp_path, mock_list_models_response):
        """
        Test that single account bypasses Circuit Breaker.
        
        What it does: Gets account when only one exists
        Purpose: Verify single account always returns (no cooldown)
        """
        print("\n=== Test: get_next_account single account bypasses Circuit Breaker ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Set failures (should be ignored for single account)
        manager._accounts[account_id].failures = 10
        manager._accounts[account_id].last_failure_time = time.time()
        
        # Act
        account = await manager.get_next_account("claude-opus-4.5")
        
        # Assert
        print(f"Got account: {account is not None}")
        assert account is not None  # Single account always returns


class TestAccountManagerReportSuccess:
    """
    Tests for AccountManager.report_success() method.
    """
    
    @pytest.mark.asyncio
    async def test_report_success_reset_failures(self, tmp_path, mock_list_models_response):
        """
        Test that report_success resets failures to 0.
        
        What it does: Reports success after failures
        Purpose: Verify failure counter reset
        """
        print("\n=== Test: report_success resets failures ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Set failures
        manager._accounts[account_id].failures = 5
        
        # Act
        await manager.report_success(account_id, "claude-opus-4.5")
        
        # Assert
        print(f"Failures after success: {manager._accounts[account_id].failures}")
        assert manager._accounts[account_id].failures == 0
    
    @pytest.mark.asyncio
    async def test_report_success_update_stats(self, tmp_path, mock_list_models_response):
        """
        Test that report_success updates statistics.
        
        What it does: Reports success and checks stats
        Purpose: Verify statistics tracking
        """
        print("\n=== Test: report_success updates stats ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        await manager.report_success(account_id, "claude-opus-4.5")
        
        # Assert
        stats = manager._accounts[account_id].stats
        print(f"Stats: total={stats.total_requests}, successful={stats.successful_requests}")
        assert stats.total_requests == 1
        assert stats.successful_requests == 1


class TestAccountManagerReportFailure:
    """
    Tests for AccountManager.report_failure() method.
    """
    
    @pytest.mark.asyncio
    async def test_report_failure_recoverable_increment_failures(self, tmp_path, mock_list_models_response):
        """
        Test that RECOVERABLE errors increment failures.
        
        What it does: Reports RECOVERABLE failure
        Purpose: Verify failure counter increment
        """
        print("\n=== Test: report_failure RECOVERABLE increments failures ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        await manager.report_failure(
            account_id, "claude-opus-4.5",
            ErrorType.RECOVERABLE, 429, None
        )
        
        # Assert
        print(f"Failures: {manager._accounts[account_id].failures}")
        assert manager._accounts[account_id].failures == 1
    
    @pytest.mark.asyncio
    async def test_report_failure_fatal_no_increment(self, tmp_path, mock_list_models_response):
        """
        Test that FATAL errors do NOT increment failures.
        
        What it does: Reports FATAL failure
        Purpose: Verify failures not incremented for request errors
        """
        print("\n=== Test: report_failure FATAL does not increment failures ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        await manager.report_failure(
            account_id, "claude-opus-4.5",
            ErrorType.FATAL, 400, "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
        )
        
        # Assert
        print(f"Failures: {manager._accounts[account_id].failures}")
        assert manager._accounts[account_id].failures == 0  # Not incremented


class TestAccountManagerSaveState:
    """
    Tests for AccountManager._save_state() and save_state_periodically().
    """
    
    @pytest.mark.asyncio
    async def test_save_state_atomic_write(self, tmp_path):
        """
        Test atomic state saving via tmp file.
        
        What it does: Saves state and checks tmp file usage
        Purpose: Verify atomic write pattern
        """
        print("\n=== Test: save_state atomic write ===")
        
        # Arrange
        state_file = tmp_path / "state.json"
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )
        
        # Act
        await manager._save_state()
        
        # Assert
        print(f"State file exists: {state_file.exists()}")
        assert state_file.exists()
        
        # Verify tmp file was cleaned up
        tmp_file = tmp_path / "state.json.tmp"
        print(f"Tmp file exists: {tmp_file.exists()}")
        assert not tmp_file.exists()


class TestAccountManagerGetFirstAccount:
    """
    Tests for AccountManager.get_first_account() method.
    """
    
    @pytest.mark.asyncio
    async def test_get_first_account_success(self, tmp_path, mock_list_models_response):
        """
        Test getting first initialized account.
        
        What it does: Gets first account for legacy mode
        Purpose: Verify legacy mode support
        """
        print("\n=== Test: get_first_account success ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        account = manager.get_first_account()
        
        # Assert
        print(f"Got account: {account is not None}")
        assert account is not None
        assert account.auth_manager is not None
    
    def test_get_first_account_no_initialized(self, tmp_path):
        """
        Test RuntimeError when no initialized accounts.
        
        What it does: Attempts to get account when none initialized
        Purpose: Verify error handling
        """
        print("\n=== Test: get_first_account with no initialized accounts ===")
        
        # Arrange
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="No initialized accounts available"):
            manager.get_first_account()


class TestAccountManagerGetAllAvailableModels:
    """
    Tests for AccountManager.get_all_available_models() method.
    """
    
    @pytest.mark.asyncio
    async def test_get_all_available_models_collect_from_all(self, tmp_path, mock_list_models_response):
        """
        Test collecting unique models from all accounts.
        
        What it does: Gets models from multiple accounts
        Purpose: Verify model aggregation for /v1/models endpoint
        """
        print("\n=== Test: get_all_available_models collects from all ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        models = manager.get_all_available_models()
        
        # Assert
        print(f"Available models: {len(models)}")
        assert len(models) > 0
        assert isinstance(models, list)
        assert all(isinstance(m, str) for m in models)


class TestFormatDuration:
    """
    Tests for _format_duration() helper function.
    """
    
    def test_format_duration_seconds(self):
        """Test formatting seconds."""
        assert _format_duration(30) == "30s"
        assert _format_duration(59) == "59s"
    
    def test_format_duration_minutes(self):
        """Test formatting minutes."""
        assert _format_duration(60) == "1m"
        assert _format_duration(300) == "5m"
        assert _format_duration(3599) == "59m"
    
    def test_format_duration_hours(self):
        """Test formatting hours."""
        assert _format_duration(3600) == "1h"
        assert _format_duration(7200) == "2h"
        assert _format_duration(86399) == "23h"
    
    def test_format_duration_days(self):
        """Test formatting days."""
        assert _format_duration(86400) == "1d"
        assert _format_duration(172800) == "2d"


# =============================================================================
# Live model discovery wiring (management endpoint)
# =============================================================================

# Real model IDs returned by management.{region}.kiro.dev/ListAvailableModels
DISCOVERED_MODEL_IDS = [
    "auto",
    "claude-sonnet-5",
    "claude-opus-4.8",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "claude-opus-4.7",
    "claude-opus-4.6",
    "claude-sonnet-4.6",
    "claude-opus-4.5",
    "claude-sonnet-4.5",
    "claude-sonnet-4",
    "claude-haiku-4.5",
    "deepseek-3.2",
    "minimax-m2.5",
    "minimax-m2.1",
    "glm-5",
    "qwen3-coder-next",
]


def _discovery_body(model_ids):
    """
    Build a /ListAvailableModels response body in the real upstream shape.

    Args:
        model_ids: Model IDs to include

    Returns:
        Response body dictionary
    """
    return {
        "models": [
            {
                "description": f"Description of {model_id}",
                "modelId": model_id,
                "modelName": model_id,
                "promptCaching": {"supportsPromptCaching": True},
                "rateMultiplier": 1.0,
                "rateUnit": "Credit",
                "supportedInputTypes": ["TEXT"],
            }
            for model_id in model_ids
        ]
    }


class FakeDiscoveryClient:
    """
    Minimal stand-in for httpx.AsyncClient used by model discovery.

    Records real httpx.Request objects so query and header assembly is still
    performed by httpx, while no traffic ever leaves the process.

    Attributes:
        requests: Every request issued, in order
        closed: True once aclose() was awaited
    """

    def __init__(self, responder):
        """
        Initialize the fake client.

        Args:
            responder: Callable receiving (request, index) that returns an
                httpx.Response or raises to simulate a transport failure
        """
        self.requests = []
        self.closed = False
        self._responder = responder

    async def get(self, url, params=None, headers=None):
        """Record the request and return the configured response."""
        request = httpx.Request("GET", url, params=params, headers=headers)
        index = len(self.requests)
        self.requests.append(request)

        response = self._responder(request, index)
        response.request = request
        return response

    async def aclose(self):
        """Mark the client as closed."""
        self.closed = True


def _make_runtime_account(tmp_path, region="us-east-1"):
    """
    Create an AccountManager with one JSON account on the runtime endpoint.

    Args:
        tmp_path: pytest temporary directory
        region: Region stored in the credentials file

    Returns:
        Tuple of (manager, account_id)
    """
    test_json = tmp_path / "runtime_account.json"
    test_json.write_text(json.dumps({
        "refreshToken": "discovery_refresh",
        "accessToken": "discovery_access",
        "expiresAt": "2099-01-01T00:00:00.000Z",
        "profileArn": "arn:aws:codewhisperer:us-east-1:123456789012:profile/DISCOVERY",
        "region": region
    }))

    creds_file = tmp_path / "credentials.json"
    creds_file.write_text(json.dumps([
        {"type": "json", "path": str(test_json), "enabled": True}
    ]))

    manager = AccountManager(
        credentials_file=str(creds_file),
        state_file=str(tmp_path / "state.json")
    )

    return manager, str(test_json.resolve())


class TestAccountManagerModelDiscovery:
    """
    Tests for live model discovery wiring in AccountManager.

    Runtime-endpoint accounts must discover their real model list from the
    management host and degrade to FALLBACK_MODELS without breaking startup.
    """

    @pytest.mark.asyncio
    async def test_runtime_account_discovers_real_model_list(self, tmp_path):
        """
        Test that a runtime-endpoint account gets the discovered model list.

        What it does: Initializes an account with a mocked management response
        Purpose: Verify the 18 real models replace the 13-model static fallback
        """
        print("\n=== Test: runtime account discovers real model list ===")

        manager, account_id = _make_runtime_account(tmp_path)
        await manager.load_credentials()

        fake_client = FakeDiscoveryClient(
            lambda request, index: httpx.Response(
                200, json=_discovery_body(DISCOVERED_MODEL_IDS)
            )
        )

        with patch("kiro.model_discovery.httpx.AsyncClient", return_value=fake_client):
            success = await manager._initialize_account(account_id)

        account = manager._accounts[account_id]
        cached_ids = account.model_cache.get_all_model_ids()
        print(f"Success: {success}, cached models: {len(cached_ids)}")

        assert success is True
        assert sorted(cached_ids) == sorted(DISCOVERED_MODEL_IDS)

        for new_model in ("gpt-5.6-sol", "claude-sonnet-5", "claude-opus-4.8"):
            print(f"Checking {new_model} is valid...")
            assert account.model_cache.is_valid_model(new_model) is True

        assert fake_client.closed is True

    @pytest.mark.asyncio
    async def test_discovery_request_targets_management_host(self, tmp_path):
        """
        Test the outgoing discovery request shape.

        What it does: Inspects the recorded request from account initialization
        Purpose: Lock the verified endpoint contract (GET, origin, profileArn)
        """
        print("\n=== Test: discovery request targets management host ===")

        manager, account_id = _make_runtime_account(tmp_path)
        await manager.load_credentials()

        fake_client = FakeDiscoveryClient(
            lambda request, index: httpx.Response(200, json=_discovery_body(["auto"]))
        )

        with patch("kiro.model_discovery.httpx.AsyncClient", return_value=fake_client):
            await manager._initialize_account(account_id)

        assert len(fake_client.requests) == 1
        request = fake_client.requests[0]
        print(f"{request.method} {request.url}")

        assert request.method == "GET"
        assert request.url.host == "management.us-east-1.kiro.dev"
        assert request.url.path == "/ListAvailableModels"
        assert request.url.params["origin"] == "AI_EDITOR"
        assert request.url.params["profileArn"].startswith("arn:aws:codewhisperer:")
        assert request.headers["content-type"] == "application/json"
        assert "x-amz-target" not in request.headers
        assert "KiroIDE" in request.headers["user-agent"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [403, 404, 500])
    async def test_http_failure_falls_back_to_static_models(self, tmp_path, status_code):
        """
        Test fallback to FALLBACK_MODELS on HTTP errors.

        What it does: Returns an error status from the management host
        Purpose: Startup must never break because of model discovery
        """
        print(f"\n=== Test: HTTP {status_code} falls back to static models ===")

        manager, account_id = _make_runtime_account(tmp_path)
        await manager.load_credentials()

        fake_client = FakeDiscoveryClient(
            lambda request, index: httpx.Response(status_code, json={"message": "error"})
        )

        with patch("kiro.model_discovery.httpx.AsyncClient", return_value=fake_client):
            success = await manager._initialize_account(account_id)

        account = manager._accounts[account_id]
        cached_ids = sorted(account.model_cache.get_all_model_ids())
        expected_ids = sorted(m["modelId"] for m in FALLBACK_MODELS)
        print(f"Success: {success}, cached: {len(cached_ids)} models")

        assert success is True
        assert cached_ids == expected_ids

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exception_factory",
        [
            lambda request: httpx.ReadTimeout("slow", request=request),
            lambda request: httpx.ConnectError("dns", request=request),
        ],
        ids=["timeout", "network_error"],
    )
    async def test_transport_failure_falls_back_to_static_models(
        self, tmp_path, exception_factory
    ):
        """
        Test fallback on timeouts and network errors.

        What it does: Raises transport errors from the fake client
        Purpose: Offline or blocked networks must still start the gateway
        """
        print("\n=== Test: transport failure falls back to static models ===")

        def responder(request, index):
            raise exception_factory(request)

        manager, account_id = _make_runtime_account(tmp_path)
        await manager.load_credentials()

        fake_client = FakeDiscoveryClient(responder)

        with patch("kiro.model_discovery.httpx.AsyncClient", return_value=fake_client):
            success = await manager._initialize_account(account_id)

        account = manager._accounts[account_id]
        print(f"Success: {success}, cached: {account.model_cache.size} models")

        assert success is True
        assert account.model_cache.size == len(FALLBACK_MODELS)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body,raw",
        [
            (None, "<html>not json</html>"),
            ({"other": "field"}, None),
            ({"models": []}, None),
            ({"models": [{"modelName": "no id"}]}, None),
        ],
        ids=["malformed_json", "models_missing", "models_empty", "entries_without_id"],
    )
    async def test_unusable_body_falls_back_to_static_models(self, tmp_path, body, raw):
        """
        Test fallback when the response body carries no usable models.

        What it does: Returns malformed or empty bodies from the management host
        Purpose: Clients must never see an empty model list
        """
        print("\n=== Test: unusable body falls back to static models ===")

        def responder(request, index):
            if raw is not None:
                return httpx.Response(200, text=raw)
            return httpx.Response(200, json=body)

        manager, account_id = _make_runtime_account(tmp_path)
        await manager.load_credentials()

        fake_client = FakeDiscoveryClient(responder)

        with patch("kiro.model_discovery.httpx.AsyncClient", return_value=fake_client):
            success = await manager._initialize_account(account_id)

        account = manager._accounts[account_id]
        print(f"Success: {success}, cached: {account.model_cache.size} models")

        assert success is True
        assert account.model_cache.size == len(FALLBACK_MODELS)

    @pytest.mark.asyncio
    async def test_discovery_paginates_across_pages(self, tmp_path):
        """
        Test that paginated discovery results are merged during initialization.

        What it does: Serves two pages joined by nextToken
        Purpose: A truncated list would hide models from clients
        """
        print("\n=== Test: discovery paginates across pages ===")

        page_one = _discovery_body(DISCOVERED_MODEL_IDS[:9])
        page_one["nextToken"] = "page-2"
        page_two = _discovery_body(DISCOVERED_MODEL_IDS[9:])

        def responder(request, index):
            return httpx.Response(200, json=page_one if index == 0 else page_two)

        manager, account_id = _make_runtime_account(tmp_path)
        await manager.load_credentials()

        fake_client = FakeDiscoveryClient(responder)

        with patch("kiro.model_discovery.httpx.AsyncClient", return_value=fake_client):
            await manager._initialize_account(account_id)

        account = manager._accounts[account_id]
        print(f"Requests: {len(fake_client.requests)}, cached: {account.model_cache.size}")

        assert len(fake_client.requests) == 2
        assert account.model_cache.size == len(DISCOVERED_MODEL_IDS)

    @pytest.mark.asyncio
    async def test_non_runtime_endpoint_keeps_existing_behavior(
        self, tmp_path, mock_list_models_response
    ):
        """
        Test that non-runtime accounts still use the q host path.

        What it does: Forces _is_runtime_endpoint to False
        Purpose: The legacy endpoint behavior must not regress
        """
        print("\n=== Test: non-runtime endpoint keeps existing behavior ===")

        manager, account_id = _make_runtime_account(tmp_path)
        await manager.load_credentials()

        with patch("kiro.account_manager._is_runtime_endpoint", return_value=False), \
             patch("kiro.account_manager.discover_models_with_fallback") as mock_discovery, \
             patch("kiro.account_manager.KiroHttpClient") as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client

            success = await manager._initialize_account(account_id)

        account = manager._accounts[account_id]
        cached_ids = sorted(account.model_cache.get_all_model_ids())
        print(f"Success: {success}, cached: {cached_ids}")

        assert success is True
        assert mock_discovery.called is False
        assert mock_client.request_with_retry.await_count == 1
        assert cached_ids == sorted(m["modelId"] for m in mock_list_models_response["models"])

    @pytest.mark.asyncio
    async def test_discovered_models_flow_to_both_api_surfaces(self, tmp_path):
        """
        Test that discovered models reach the resolver and the aggregate list.

        What it does: Reads ModelResolver.get_available_models() and
            AccountManager.get_all_available_models() after discovery
        Purpose: /v1/models (OpenAI) and the Anthropic surface share this data
        """
        print("\n=== Test: discovered models flow to both API surfaces ===")

        manager, account_id = _make_runtime_account(tmp_path)
        await manager.load_credentials()

        fake_client = FakeDiscoveryClient(
            lambda request, index: httpx.Response(
                200, json=_discovery_body(DISCOVERED_MODEL_IDS)
            )
        )

        with patch("kiro.model_discovery.httpx.AsyncClient", return_value=fake_client):
            await manager._initialize_account(account_id)

        account = manager._accounts[account_id]
        resolver_models = account.model_resolver.get_available_models()
        aggregate_models = manager.get_all_available_models()

        print(f"Resolver models: {resolver_models}")
        print(f"Aggregate models: {aggregate_models}")

        for model_id in ("gpt-5.6-sol", "claude-sonnet-5", "claude-opus-4.8", "glm-5"):
            assert model_id in resolver_models
            assert model_id in aggregate_models

        # Both surfaces resolve the new models through the same cache
        resolution = account.model_resolver.resolve("claude-sonnet-5")
        print(f"Resolution: {resolution}")
        assert resolution.internal_id == "claude-sonnet-5"
        assert resolution.source == "cache"
        assert resolution.is_verified is True

    @pytest.mark.asyncio
    async def test_presentation_keeps_auto_kiro_alias_and_hides_auto(self, tmp_path):
        """
        Test that model list presentation is unchanged after discovery.

        What it does: Checks the exposed model list for auto / auto-kiro
        Purpose: /v1/models must keep showing the alias, not the bare "auto"
        """
        print("\n=== Test: presentation keeps auto-kiro and hides auto ===")

        manager, account_id = _make_runtime_account(tmp_path)
        await manager.load_credentials()

        fake_client = FakeDiscoveryClient(
            lambda request, index: httpx.Response(
                200, json=_discovery_body(DISCOVERED_MODEL_IDS)
            )
        )

        with patch("kiro.model_discovery.httpx.AsyncClient", return_value=fake_client):
            await manager._initialize_account(account_id)

        account = manager._accounts[account_id]
        exposed = account.model_resolver.get_available_models()
        aggregate = manager.get_all_available_models()

        print(f"Exposed models: {exposed}")
        assert "auto-kiro" in exposed
        assert "auto" not in exposed
        assert "auto-kiro" in aggregate
        assert "auto" not in aggregate

        # "auto" still works when requested explicitly (hidden, not removed)
        assert account.model_cache.is_valid_model("auto") is True
        assert account.model_resolver.resolve("auto-kiro").internal_id == "auto"

    @pytest.mark.asyncio
    async def test_ttl_refresh_uses_management_discovery(self, tmp_path):
        """
        Test that the TTL refresh path also discovers live models.

        What it does: Initializes with a small list, then refreshes with a larger one
        Purpose: New models must appear without restarting the gateway
        """
        print("\n=== Test: TTL refresh uses management discovery ===")

        manager, account_id = _make_runtime_account(tmp_path)
        await manager.load_credentials()

        initial_client = FakeDiscoveryClient(
            lambda request, index: httpx.Response(200, json=_discovery_body(["auto", "glm-5"]))
        )
        with patch("kiro.model_discovery.httpx.AsyncClient", return_value=initial_client):
            await manager._initialize_account(account_id)

        account = manager._accounts[account_id]
        print(f"Cached after init: {account.model_cache.size}")
        assert account.model_cache.size == 2

        refresh_client = FakeDiscoveryClient(
            lambda request, index: httpx.Response(
                200, json=_discovery_body(DISCOVERED_MODEL_IDS)
            )
        )
        with patch("kiro.model_discovery.httpx.AsyncClient", return_value=refresh_client):
            await manager._refresh_account_models(account_id)

        print(f"Cached after refresh: {account.model_cache.size}")
        assert account.model_cache.size == len(DISCOVERED_MODEL_IDS)
        assert account.model_cache.is_valid_model("gpt-5.6-sol") is True
        assert "gpt-5.6-sol" in manager._model_to_accounts

    @pytest.mark.asyncio
    async def test_ttl_refresh_keeps_richer_cache_when_discovery_fails(self, tmp_path):
        """
        Test that a failed refresh does not downgrade a richer cache.

        What it does: Refreshes after a successful discovery, but with a failing endpoint
        Purpose: A temporary outage must not remove models clients already use
        """
        print("\n=== Test: failed refresh keeps richer cache ===")

        manager, account_id = _make_runtime_account(tmp_path)
        await manager.load_credentials()

        initial_client = FakeDiscoveryClient(
            lambda request, index: httpx.Response(
                200, json=_discovery_body(DISCOVERED_MODEL_IDS)
            )
        )
        with patch("kiro.model_discovery.httpx.AsyncClient", return_value=initial_client):
            await manager._initialize_account(account_id)

        account = manager._accounts[account_id]
        assert account.model_cache.size == len(DISCOVERED_MODEL_IDS)

        failing_client = FakeDiscoveryClient(
            lambda request, index: httpx.Response(500, json={"message": "error"})
        )
        with patch("kiro.model_discovery.httpx.AsyncClient", return_value=failing_client):
            await manager._refresh_account_models(account_id)

        print(f"Cached after failed refresh: {account.model_cache.size}")
        assert account.model_cache.size == len(DISCOVERED_MODEL_IDS)
        assert account.model_cache.is_valid_model("claude-sonnet-5") is True

    @pytest.mark.asyncio
    async def test_ttl_refresh_falls_back_when_cache_is_not_richer(self, tmp_path):
        """
        Test that a failed refresh still yields the static list for a small cache.

        What it does: Starts with 2 discovered models, then fails the refresh
        Purpose: Keep the documented safety net when nothing better is cached
        """
        print("\n=== Test: failed refresh falls back for small cache ===")

        manager, account_id = _make_runtime_account(tmp_path)
        await manager.load_credentials()

        initial_client = FakeDiscoveryClient(
            lambda request, index: httpx.Response(200, json=_discovery_body(["auto", "glm-5"]))
        )
        with patch("kiro.model_discovery.httpx.AsyncClient", return_value=initial_client):
            await manager._initialize_account(account_id)

        account = manager._accounts[account_id]
        assert account.model_cache.size == 2

        failing_client = FakeDiscoveryClient(
            lambda request, index: httpx.Response(500, json={"message": "error"})
        )
        with patch("kiro.model_discovery.httpx.AsyncClient", return_value=failing_client):
            await manager._refresh_account_models(account_id)

        cached_ids = sorted(account.model_cache.get_all_model_ids())
        print(f"Cached after failed refresh: {cached_ids}")
        assert cached_ids == sorted(m["modelId"] for m in FALLBACK_MODELS)

    @pytest.mark.asyncio
    async def test_discovered_models_have_usable_token_limits(self, tmp_path):
        """
        Test token limit defaults for discovered models.

        What it does: Reads get_max_input_tokens() for every discovered model
        Purpose: The real payload has no token limit fields - no KeyError / None math
        """
        print("\n=== Test: discovered models have usable token limits ===")

        manager, account_id = _make_runtime_account(tmp_path)
        await manager.load_credentials()

        fake_client = FakeDiscoveryClient(
            lambda request, index: httpx.Response(
                200, json=_discovery_body(DISCOVERED_MODEL_IDS)
            )
        )

        with patch("kiro.model_discovery.httpx.AsyncClient", return_value=fake_client):
            await manager._initialize_account(account_id)

        account = manager._accounts[account_id]
        for model_id in DISCOVERED_MODEL_IDS:
            max_tokens = account.model_cache.get_max_input_tokens(model_id)
            print(f"{model_id}: {max_tokens}")
            assert isinstance(max_tokens, int)
            assert max_tokens > 0
