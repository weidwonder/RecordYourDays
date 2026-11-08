# Agent instructions / 给 AI agent 的说明

This project turns a day of audio recordings into speaker-labeled transcripts, then into a diary.
本项目把一整天的录音转成带说话人标签的转写稿，再写成日记。

- When the user says "write yesterday's diary" / 用户说“写昨天的日记”时：follow [docs/diary-workflow.en.md](docs/diary-workflow.en.md) / 照 [docs/diary-workflow.md](docs/diary-workflow.md) 做。
- Tool usage / 工具用法：see [README.md](README.md) / [README.zh-CN.md](README.zh-CN.md).
- Privacy / 隐私：`raw/`, `transcripts/`, `diary/`, `cache/`, `memory/` never leave this machine (and your own GPU host) and are never committed. 录音、转写稿、日记、记忆只在本机和你自己的 GPU 主机上处理，不提交 git。
- Long-term facts about the user go in `memory/` (one fact per file, indexed by `memory/MEMORY.md`). If it exists, read it: @memory/MEMORY.md
