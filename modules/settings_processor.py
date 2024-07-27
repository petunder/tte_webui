# modules/settings_processor.py
from classes.settings import Settings

settings = Settings()

def get_all_settings():
    return settings.settings

def update_settings(new_settings):
    for key, value in new_settings.items():
        settings.update_setting(key, value)
    settings.save_settings()
    return settings.settings

def reset_settings():
    settings.reset_to_default()
    settings.save_settings()
    return settings.settings
