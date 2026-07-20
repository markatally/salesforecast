"""
日志工具模块

提供统一的日志记录功能，支持控制台和文件输出
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# ============ 全局配置 ============
_GLOBAL_LOGGER_NAME = None  # 全局logger名称
_GLOBAL_LOGGER_termANCE = None  # 全局logger实例


def setup_global_logger(
    name: str = "SellinForecast",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    设置全局日志记录器

    Args:
        name: 日志记录器名称
        log_dir: 日志文件目录（仅当 log_file 未指定时使用）
        log_file: 日志文件名
        level: 日志级别
        console_output: 是否输出到控制台
        file_output: 是否输出到文件

    Returns:
        配置好的日志记录器

    Example:
        >>> # 在 main.py 中配置全局logger
        >>> setup_global_logger("MyApp", log_dir="logs")
        >>>
        >>> # 其他模块直接使用，无需重新配置
        >>> logger = get_global_logger()
        >>> logger.info("Hello")
    """
    global _GLOBAL_LOGGER_NAME, _GLOBAL_LOGGER_termANCE

    _GLOBAL_LOGGER_NAME = name
    _GLOBAL_LOGGER_termANCE = _setup_logger(
        name=name,
        log_dir=log_dir,
        log_file=log_file,
        level=level,
        console_output=console_output,
        file_output=file_output
    )

    return _GLOBAL_LOGGER_termANCE


def get_global_logger() -> logging.Logger:
    """
    获取全局日志记录器

    Returns:
        全局日志记录器实例

    Example:
        >>> # 在任意模块中使用
        >>> logger = get_global_logger()
        >>> logger.info("This message uses the global logger")
        >>> logger.warning("Warning message")
        >>> logger.error("Error message")
    """
    global _GLOBAL_LOGGER_termANCE

    if _GLOBAL_LOGGER_termANCE is None:
        # 如果还未设置全局logger，使用默认配置创建
        _GLOBAL_LOGGER_termANCE = setup_global_logger()

    return _GLOBAL_LOGGER_termANCE


def _setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    内部方法：设置日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 防止重复添加处理器
    if logger.handlers:
        return logger

    # 日志格式
    formatter = logging.Formatter(
        f'%(asctime)s - %(name)s - %(module)s - %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    # 文件处理器
    if file_output:
        # 确定日志文件是否存在，如果不存在则创建
        if log_file is None:
            # 确定日志目录
            if log_dir is None:
                log_dir = Path(__file__).parent.parent.parent / 'logs'
            else:
                log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            # 创建日志文件，文件名包含时间戳
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f'{name}_{timestamp}.log'

        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger

logger = get_global_logger()