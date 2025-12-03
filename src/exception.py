import sys
from src.logger import logging


def error_message_detail(error, error_detail: sys):
    # Call exc_info() as a method
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    # Use f-string for clarity and correct placeholder indexing
    error_message = f"Error occurred in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # Correctly call the parent class's __init__ method
        super().__init__(error_message)

        # Pass the original error_message (exception object) and sys
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    # Correct the dunder method to __str__ for string representation
    def __str__(self):
        return self.error_message


# Example of how to use it:

# if __name__ == "__main__":
#     try : 
#         a= 1/0
#     except Exception as e:
#         logging.info("DIVIDING BY ZERO ERROR")
#         raise CustomException(e,sys)