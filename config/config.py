import toml

with open("D:\XM\Evaluation\config\config.toml", "r", encoding="utf-8") as f:
    config = toml.load(f)

API_CONFIG = config.get("api", {})

PATH_CONFIG = config.get("path", {})

MODEL_CONFIG = config.get("model", {})

DATABASE_CONFIG = config.get("database", {})

PATH_CONFIG = config.get("path", {})

