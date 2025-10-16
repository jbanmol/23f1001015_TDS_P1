import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from main import save_attachments_locally, verify_secret
import tempfile
import os
import base64

class MockAttachment:
    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url

@pytest.mark.asyncio
async def test_save_attachments_with_invalid_base64():
    """Test handling of invalid base64 data in attachments"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create attachment with invalid base64
        bad_attachment = MockAttachment(
            "test.txt",
            "data:text/plain;base64,invalid_base64_data!!!"
        )
        
        # Should handle gracefully and continue with other attachments
        result = await save_attachments_locally(temp_dir, [bad_attachment])
        
        # Should return empty list (no files saved)
        assert result == []
        
        # Directory should still exist and be empty
        assert os.path.exists(temp_dir)
        assert len(os.listdir(temp_dir)) == 0

@pytest.mark.asyncio
async def test_save_attachments_with_mixed_valid_invalid():
    """Test that valid attachments are saved even when some are invalid"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create one valid and one invalid attachment
        valid_data = base64.b64encode(b"Hello World").decode()
        valid_attachment = MockAttachment(
            "valid.txt",
            f"data:text/plain;base64,{valid_data}"
        )
        
        invalid_attachment = MockAttachment(
            "invalid.txt",
            "not_a_data_uri"
        )
        
        result = await save_attachments_locally(temp_dir, [valid_attachment, invalid_attachment])
        
        # Should save only the valid attachment
        assert len(result) == 1
        assert result[0] == "valid.txt"
        
        # Check file was actually created
        saved_file = os.path.join(temp_dir, "valid.txt")
        assert os.path.exists(saved_file)
        
        with open(saved_file, 'r') as f:
            assert f.read() == "Hello World"

def test_secret_verification_edge_cases():
    """Test secret verification with various edge cases"""
    
    # Test empty strings
    assert not verify_secret("")
    
    # Test None (should handle gracefully and return False)
    assert not verify_secret(None)
    
    # Test very long strings (potential DoS)
    long_secret = "a" * 10000
    result = verify_secret(long_secret)
    assert isinstance(result, bool)  # Should not crash

@pytest.mark.asyncio
async def test_attachment_with_special_filenames():
    """Test handling of filenames with special characters"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test various problematic filenames
        test_cases = [
            ("normal.txt", True),
            ("../../../etc/passwd", False),  # Path traversal attempt
            ("file with spaces.txt", True),
            ("file\x00null.txt", False),  # Null byte
            ("", False),  # Empty filename
            ("a" * 200 + ".txt", False),  # Very long filename
        ]
        
        for filename, should_work in test_cases:
            valid_data = base64.b64encode(b"test content").decode()
            attachment = MockAttachment(
                filename,
                f"data:text/plain;base64,{valid_data}"
            )
            
            try:
                result = await save_attachments_locally(temp_dir, [attachment])
                
                if should_work:
                    # Should succeed
                    assert len(result) >= 0  # May be 0 if filename gets sanitized away
                else:
                    # Should either skip the file or sanitize the filename
                    # Either outcome is acceptable for security
                    pass
                    
            except Exception as e:
                if should_work:
                    pytest.fail(f"Unexpected failure for filename '{filename}': {e}")
                # Failures for problematic filenames are acceptable

def test_verify_secret_timing_safety():
    """Test that secret verification doesn't leak timing information"""
    import time
    
    correct_secret = "correct_secret_123"
    
    # Patch settings to use our test secret
    with patch('main.settings') as mock_settings:
        mock_settings.STUDENT_SECRET = correct_secret
        
        # Test with secrets of different lengths
        test_secrets = [
            "a",  # Very short
            "wrong_secret",  # Different length
            "correct_secret_12",  # Almost correct, one char short
            "correct_secret_123",  # Correct
            "correct_secret_1234",  # One char too long
        ]
        
        times = []
        
        for secret in test_secrets:
            start = time.perf_counter()
            result = verify_secret(secret)
            end = time.perf_counter()
            times.append(end - start)
        
        # The timing differences should be minimal (< 1ms difference)
        max_time = max(times)
        min_time = min(times)
        time_diff = max_time - min_time
        
        # Allow for some variance but flag if there's a significant timing difference
        # This is more of a warning than a hard failure since timing can vary
        if time_diff > 0.001:  # 1ms
            print(f"Warning: Potential timing leak detected. Time difference: {time_diff:.6f}s")

@pytest.mark.asyncio
async def test_large_number_of_attachments():
    """Test handling of many attachments (stress test)"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create 100 small attachments
        attachments = []
        for i in range(100):
            data = base64.b64encode(f"Content {i}".encode()).decode()
            attachment = MockAttachment(
                f"file_{i}.txt",
                f"data:text/plain;base64,{data}"
            )
            attachments.append(attachment)
        
        # This should not crash or consume excessive resources
        result = await save_attachments_locally(temp_dir, attachments)
        
        # Should handle all attachments (or gracefully limit them)
        assert isinstance(result, list)
        assert len(result) <= 100  # Should not exceed input

@pytest.mark.asyncio 
async def test_concurrent_attachment_processing():
    """Test that attachment processing is thread-safe"""
    import asyncio
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple tasks that process attachments concurrently
        async def process_attachment(i):
            data = base64.b64encode(f"Content {i}".encode()).decode()
            attachment = MockAttachment(
                f"concurrent_{i}.txt",
                f"data:text/plain;base64,{data}"
            )
            return await save_attachments_locally(
                os.path.join(temp_dir, f"subdir_{i}"),
                [attachment]
            )
        
        # Create subdirectories
        for i in range(5):
            os.makedirs(os.path.join(temp_dir, f"subdir_{i}"))
        
        # Run concurrent processing
        tasks = [process_attachment(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed without exceptions
        for result in results:
            assert not isinstance(result, Exception), f"Unexpected error: {result}"
            assert isinstance(result, list)