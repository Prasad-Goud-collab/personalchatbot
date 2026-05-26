from configparser import ConfigParser
from pathlib import Path


class Config:
    def __init__(self, config_file: str | Path | None = None):
        if config_file is None:
            config_file = Path(__file__).resolve().parent / "uiconfig.ini"
        self.config = ConfigParser()
        self.config.read(config_file)

    def get_llm_options(self):
        return self.config["DEFAULT"].get("LLM_OPTIONS", "").split(", ")

    def get_usecase_options(self):
        return self.config["DEFAULT"].get("USECASE_OPTIONS", "").split(",")

    def get_groq_model_options(self):
        return self.config["DEFAULT"].get("GROQ_MODEL_OPTIONS", "").split(",")

    def get_page_title(self):
        return self.config["DEFAULT"].get("PAGE_TITLE", "")


__all__ = ["Config"]