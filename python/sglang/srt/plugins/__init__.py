"""
SGLang Unified Plugin Framework.

Supports two types of plugins via setuptools entry_points:
1. Hardware Platform Plugins (sglang.srt.platforms) - register custom hardware platforms
2. General Plugins (sglang.srt.plugins) - inject hooks into functions/methods, replace classes, etc.

Plugins are discovered automatically when installed via pip.
- Platform plugins: use ``SGLANG_PLATFORM`` to select when multiple are installed.
- General plugins: use ``SGLANG_PLUGINS`` (comma-separated) to restrict which are loaded.


NOTE 示例

一套文件：
demo/
├─ pyproject.toml
└─ demo_plugin/
   └─ __init__.py

pyproject.toml 配置文件
```
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "demo-plugin"
version = "0.1.0"

[project.entry-points."sglang.srt.plugins"]
hello = "demo_plugin:register"
```

demo_plugin/__init__.py：插件逻辑实现
···
def register():
    print("插件入口被调用了")
···

插件使用
···
from importlib.metadata import entry_points

for ep in entry_points(group="sglang.srt.plugins", name="hello"):
    print(ep.name)   # hello
    print(ep.value)  # demo_plugin:register

    func = ep.load() # 取得 register 函数对象
    func()          # 输出：插件入口被调用了
···

"""

import logging
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

from sglang.srt.environ import envs
from sglang.srt.plugins.hook_registry import (
    HookRegistry,
    HookSource,
    _current_plugin_source,
)

logger = logging.getLogger(__name__)

# Entry point group names
PLATFORM_PLUGINS_GROUP = "sglang.srt.platforms"
GENERAL_PLUGINS_GROUP = "sglang.srt.plugins"

# Guard against multiple loads in the same process
_plugins_loaded = False


def load_plugins_by_group(
    group: str, # tip GENERAL_PLUGINS_GROUP
    excluded_dists: set[str] | None = None, # 空
) -> dict[str, tuple[Callable[[], Any], str | None]]:
    """
    Discover and load plugins registered under the given entry point group.

    Args:
        group: The setuptools entry_point group name.
        excluded_dists: Distribution names to skip. Plugins from these
            distributions are never ``ep.load()``-ed (avoids importing
            their modules and pulling hardware-specific dependencies).

    Returns:
        Dictionary mapping plugin name to ``(callable, dist_name)``.
    """
    # SGLANG_PLUGINS whitelist (comma-separated plugin names)
    allowed_set: set[str] | None = None
    allowed_str = envs.SGLANG_PLUGINS.get()
    if allowed_str:
        allowed_set = {x.strip() for x in allowed_str.split(",") if x.strip()}

    # NOTE 获取 pyproject.toml 中配置的 plugin
    discovered = entry_points(group=group)
    if len(discovered) == 0:
        logger.debug("No plugins found for group %s.", group)
        return {}

    logger.info("Available plugins for group %s:", group)
    for ep in discovered:
        logger.info("  - %s -> %s", ep.name, ep.value)

    plugins: dict[str, tuple[Callable[[], Any], str | None]] = {}
    for ep in discovered:
        if allowed_set is not None and ep.name not in allowed_set:
            logger.info("Skipping plugin %s (not in SGLANG_PLUGINS)", ep.name)
            continue
        dist_name = ep.dist.name if ep.dist else None
        if excluded_dists and dist_name in excluded_dists:
            logger.info(
                "Skipping plugin %s (dist %s excluded by SGLANG_PLATFORM)",
                ep.name,
                dist_name,
            )
            continue
        try:
            func = ep.load()
            plugins[ep.name] = (func, dist_name)
            logger.info("Loaded plugin %s from group %s", ep.name, group)
        except Exception:
            logger.exception("Failed to load plugin %s from group %s", ep.name, group)

    return plugins


def _get_excluded_dists() -> set[str]:
    """Compute dist names to skip when ``SGLANG_PLATFORM`` is set.
    tip 未指定平台时返回空集合，不排除插件分发包

    Returns dist names that provide a platform plugin but are NOT the one
    selected by ``SGLANG_PLATFORM``.  This prevents unselected platform
    packages from registering hooks that pull their hardware dependencies.
    """
    selected = envs.SGLANG_PLATFORM.get()
    if not selected:
        return set()
    platform_eps = entry_points(group=PLATFORM_PLUGINS_GROUP)
    return {ep.dist.name for ep in platform_eps if ep.dist and ep.name != selected}


def load_plugins():
    """
    Load and execute all general plugins, then apply registered hooks.

    Idempotent - safe to call multiple times. General plugins are functions
    whose side effects (registering hooks, replacing classes, etc.) are the
    desired behavior. Return values are ignored.

    When ``SGLANG_PLATFORM`` is set, general plugins from unselected platform
    packages are automatically skipped (avoids pulling their dependencies).

    After all plugins execute, ``HookRegistry.apply_hooks()`` is called
    automatically so callers only need this single function call.

    This should be called early in every process (main, engine core, workers).

    NOTE hook 注册链路如下
        run_scheduler_process()  ← scheduler 进程入口
            │
            ├─ load_plugins()
            │    │
            │    ├─ ① 发现插件并注册钩子
            │    │    → load_plugins_by_group("sglang.srt.plugins")
            │    │    → entry_points(...)：发现已安装的插件
            │    │    → ep.load()：取得插件入口函数 func
            │    │    → func()：示例中为 my_general_plugin.register()
            │    │    → HookRegistry.register(target, my_hook, HookType.AROUND)
            │    │    → _hooks[target].append(...)：记录钩子 ← register 的作用
            │    │
            │    └─ ② 应用已注册的钩子
            │         → HookRegistry.apply_hooks()
            │         → HookRegistry._apply_target(target, hooks)
            │         → _wrap_fn(original, my_hook, HookType.AROUND)
            │         → 返回包装函数 wrapper ← 此时尚未执行 my_hook
            │         → setattr(Scheduler, "__init__", wrapper)：替换目标方法
            │
            └─ ③ 创建 Scheduler，触发钩子
                → Scheduler(server_args, port_args)
                → wrapper(self, server_args, port_args)
                → my_hook(original_fn, self, server_args, port_args)
                → original_fn(self, server_args, port_args)：执行原 __init__
    """
    # tip 避免当前函数多次被调用导致 hook 重复加载
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True

    # NOTE
    plugins = load_plugins_by_group(
        GENERAL_PLUGINS_GROUP,
        excluded_dists=_get_excluded_dists(),
    )

    # tip
    #
    # 对于
    #   [project.entry-points."sglang.srt.plugins"]
    #   hello = "demo_plugin:register"
    # name = hello， dist_name = demo_plugin
    for name, (func, dist_name) in plugins.items():
        source = HookSource(plugin_name=name, dist_name=dist_name)
        token = _current_plugin_source.set(source)
        try:
            func()
            logger.info("Executed general plugin: %s", name)
        except Exception:
            logger.exception("Failed to execute general plugin: %s", name)
        finally:
            _current_plugin_source.reset(token)

    # Apply all registered hooks (idempotent — already-patched targets are skipped).
    HookRegistry.apply_hooks()
