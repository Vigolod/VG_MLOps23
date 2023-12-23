from hydra.core.config_store import ConfigStore

from .config import Config


cs = ConfigStore.instance()
cs.store("basic_config", node=Config)
