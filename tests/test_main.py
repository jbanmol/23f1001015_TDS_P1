import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from main import call_llm_for_code

@pytest.mark.asyncio
async def test_call_llm_for_code_with_empty_response_retries_and_fails():
    with patch('httpx.AsyncClient') as mock_client, \
         patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:

        mock_post = AsyncMock()

        mock_response = MagicMock()
        mock_response.text = " "
        mock_response.raise_for_status = MagicMock()

        mock_post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value.post = mock_post

        with pytest.raises(Exception, match="LLM Code Generation Failure"):
            await call_llm_for_code("test prompt", "test_task")

        # Verify that the API was called 3 times (initial + 2 retries)
        assert mock_post.call_count == 3

        # Verify that sleep was called with increasing delays
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)
