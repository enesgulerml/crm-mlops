import os
import yaml
import sys
import pickle
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_config():
    """
    It reads the config/config.yaml file located in the project's root directory.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(current_dir))
        
        config_path = os.path.join(root_dir, "config", "config.yaml")
        
        logger.info(f"Reading the configuration file: {config_path}")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            content = yaml.safe_load(f)
            
        return content

    except Exception as e:
        raise CustomException(e, sys)
    
def save_object(file_path, obj):
    """
    Python objesini (Preprocessor, Model vb.) pickle olarak kaydeder.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
        logger.info(f"Obje kaydedildi: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)