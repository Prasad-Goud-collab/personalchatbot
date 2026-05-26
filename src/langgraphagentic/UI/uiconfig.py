from configparser import ConfigParser
from pathlib import Path

class Config:
    
    def _init_(self,config_file="./src/langgraphagentic/UI/uiconfig.ini"):
        self.config=ConfigParser()
        self.config.read(config_file)
        
    def get_llm_options(self):
        return self.config["DEFAULT"].get("LLM_OPTIONS", "").split(", ")

    def get_usecase_options(self):
        return self.config["DEFAULT"].get("USECASE_OPTIONS", "").split(",")

    def get_groq_model_options(self):
        return self.config["DEFAULT"].get("GROQ_MODEL_OPTIONS", "").split(",")

    def get_page_title(self):
        return self.config["DEFAULT"].get("PAGE_TITLE", "")
