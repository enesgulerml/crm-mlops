import sys
from src.utils.logger import get_logger

logger = get_logger(__name__)


def error_message_detail(error, error_detail: sys):
    """
    Extracts detailed error message including file name and line number.
    """
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = str(error)

    formatted_message = (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{line_number}] error message [{error_message}]"
    )
    return formatted_message


class CustomException(Exception):
    """
    Custom Exception class to handle errors with detailed context.
    Inherits from Python's built-in Exception class.
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)

        self.error_message = error_message_detail(error_message, error_detail=error_detail)

        logger.error(self.error_message)

    def __str__(self):
        return self.error_message