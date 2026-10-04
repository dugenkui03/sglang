# 学习辅助工具

- [学习进度统计](update_learn_progress.py)：按 [核心范围](LEARN.scope.json) 重算 [LEARN.md](../LEARN.md) 和 [方法清单](../LEARN.methods.md)。清单按核心阶段标记已读、推定已读和未读。在仓库根目录运行 `python quick_learn/update_learn_progress.py --write`；用 `--check` 检查统计一致性，运行 `python -m unittest quick_learn.test_learn_progress` 验证统计规则。
- [学习总结视频](quick_summary/README.md)：把已学核心概念、代码和调用关系整理成逐页幻灯片、讲稿、中文配音、字幕与视频。
- 学习范围以 [LEARN.md](../LEARN.md) 和当前对话为准；源码讲解与注释沿用 [快速学习.md](../快速学习.md)。
- 下次可直接告诉 agent：**“阅读 `quick_learn/quick_summary/`，把这次学过的内容整理成总结视频。”**
- 本目录保存可复用脚本、说明和通用模板；模型与依赖使用独立运行目录。每次视频制作先建立 `quick_summary/tasks/<任务ID>/TASK.md`，课程、草稿、缓存和产物均放在同一任务目录，整个 `tasks/` 排除在版本控制之外。
- 固定制作标准见 [总结规则](quick_summary/AGENTS.md)：提供保留英文术语、类名与方法名的中文口语讲稿；源码索引单独保存；重点方法各自成页；先规划章节、讲解节奏、逐页预计时长和总时长，再按实际配音校准。
- 幻灯片覆盖学习要求和历史讨论、核心实体及其关系、带方法定义的简化实现、数据流与流程图；对象之间有关系就画连线，整套使用柔和米灰、浅绿配色。
- 写稿时按句子或完整的小逻辑拆分，并显式标记配音片段；配音按片段尽量并行生成，再按原顺序拼接，复用缓存并单独重试失败片段。
