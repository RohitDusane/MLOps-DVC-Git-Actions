import sys
import traceback
from src.CreditRisk.logger import logging

class CustomException(Exception):
    def __init__(self, error_message):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message)
        logging.error(self.error_message)
    
    @staticmethod
    def get_detailed_error_message(error_message):
        exc_type, exc_obj, exc_tb = sys.exc_info()
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            return f"Error in file: {file_name}, line: {line_number} -> {error_message}"
        else:
            return f"Error: {error_message}"
    
    def __str__(self):
        return self.error_message


# ==========================
# Example Usage
# ==========================
# if __name__ == "__main__":
#     try:
#         a = 10 / 0  # Division by zero
#     except Exception as e:
#         logging.error(f"Error occurred: {e}")
#         raise CustomException(str(e))
