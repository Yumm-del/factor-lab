"""Streamlit 运行环境能力映射，保持线上界面与实际可用数据一致。"""


STYLE_MAP = {"无": "none", "行业": "industry", "行业+市值": "industry+size"}


def pool_options(ashare_available: bool) -> list[str]:
    """只有本地存在全A数据时才暴露全A选项。"""
    options = ["沪深300（300 只，快）"]
    if ashare_available:
        options.append("全 A（5000+ 只，首次加载慢）")
    return options


def neutralization_options(industry_available: bool) -> list[str]:
    """行业映射缺失时仅允许真正可执行的“无中性化”。"""
    return ["无", "行业", "行业+市值"] if industry_available else ["无"]


def normalize_style(label: str) -> str:
    """把中文UI标签转换为计算层枚举；未知标签立即失败。"""
    return STYLE_MAP[label]
