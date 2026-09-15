import asyncio
from unittest.mock import MagicMock


async def test():
    m = MagicMock()
    m.side_effect = RuntimeError('test')
    try:
        m()
    except Exception as e:
        print('Caught:', type(e).__name__, str(e))


asyncio.run(test())
