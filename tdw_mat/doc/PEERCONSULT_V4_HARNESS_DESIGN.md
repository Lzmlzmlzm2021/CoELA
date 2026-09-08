# PeerConsult V4 Harness 设计文档（已被取代）

本文原为 V4 尚未实现时的设计草案，其中包含距离/置信度排序、自然语言 publish、策略型 validator、deadline emergency fallback 等未进入当前 V4 Core 的候选方案。继续把它作为实现规范会与当前代码产生冲突。

当前唯一权威的设计、实现和跨 benchmark 复现文档为：

- [PeerConsult V4：设计、实现与跨 Benchmark 复现规范](REPRODUCTION_V4.md)

历史设计中的以下思想已经被当前规范吸收并修正：

- mechanism-heavy, policy-light；
- agent-specific Dashboard 与私有信息边界；
- persistent/suspended tasks；
- 固定 priority 作为非强制提示；
- atomic object claim 与 room exploration reservation；
- factual validator；
- minimal one-boundary loop guard；
- 原版 CoELA executor isolation；
- 不使用 peer LLM critic；
- 正常目标、运输和通信策略由局部 LLM 决定。

请勿从本文件复现旧草案，尤其不要把旧文档中的待实验策略静默加入 V4 Core。
