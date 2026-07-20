"""
配置工具模块

提供统一的配置管理功能，支持从 YAML 文件加载配置，并支持通过点号路径获取嵌套配置。
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """配置管理类"""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # 默认：src/config/config.yaml
            base = Path(__file__).resolve().parent.parent.parent
            self.config_path = base / "config" / "config.yaml"
        else:
            self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载 YAML 配置文件"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    def get(self, key_path: str, default=None):
        """
        通过点号路径获取配置值

        Args:
            key_path: 配置路径，用点分隔，如 'data_clean.tar_ym'
            default: 默认值

        Returns:
            配置值
        """
        keys = key_path.split(".")
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def __getitem__(self, key: str):
        """支持字典式访问"""
        return self.config[key]

# 创建全局配置对象
config = Config()
