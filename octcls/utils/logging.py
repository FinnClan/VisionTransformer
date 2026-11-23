import logging
from typing import Optional

_configured = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    global _configured
    if not _configured:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
        _configured = True
    return logging.getLogger(name if name else __name__)

