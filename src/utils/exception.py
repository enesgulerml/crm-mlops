import sys

def error_message_detail(error, error_detail):
    # Hata detayını alıyoruz
    _, _, exc_tb = error_detail.exc_info()
    
    # Hata hangi dosyada?
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Hata mesajını oluştur
    error_message = "Error in script: [{0}] Line: [{1}] Message: [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        # Parent class'a (Exception) mesajı gönder
        super().__init__(error_message)
        # Kendi mesajımızı oluştur
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message