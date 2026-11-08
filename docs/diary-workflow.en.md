# Diary workflow (instructions for an AI agent)

These are operating instructions for an AI coding agent (Claude Code, Codex, …). When the user says "write yesterday's diary" or "review my day", follow them. `CLAUDE.md` / `AGENTS.md` in the repo root point here. Adapt them to your own habits. 中文版：[diary-workflow.md](diary-workflow.md)。

## 1. Gather the material

1. **Transcribe**: run `npm run run` (defaults to yesterday) to get `transcripts/<date>/`. For a specific day: `npm run run -- 2026-01-01`.
2. **Check who is who**: run `npm run voice -- clusters <date>` to see when each speaker cluster talks and who it was matched to. Typical mistakes:
   - A family label showing up all through working hours usually means a colleague was matched as a family member. Enroll that cluster as the colleague with `npm run voice -- add-cluster <name> <date> <recording> <cluster>`, run `npm run voice -- check`, then relabel with `npm run transcribe -- <date> --relabel-only`.
   - When the user says in the recording "use this part for my voiceprint", enroll that time range with `npm run voice -- add <name> <recording> <start> <end>`.
   - Before enrolling a time range with `add`, check in the transcript which cluster is which: in a one-on-one conversation the cluster that talks the most is often the user. Use `--cluster` when needed.
   - `clusters`, `add-cluster`, and `add` without `--whole` need the GPU pipeline's cache. With the CPU pipeline only, pick a stretch where that person speaks alone and clearly, and enroll it with `add ... --whole` (up to 120 seconds).
   - Don't enroll people you only meet once or twice (events, strangers); they pollute the voiceprint library. Start new non-family people at a threshold of 0.6 and, after relabeling, check that they don't show up at unrelated times.
3. **Read every transcript**. If they are long, read them in chunks and skip unrelated audio (announcements, videos playing). The user often talks to the agent directly in the recording (questions, requests); do what they ask. It helps to agree on a trigger word (e.g. "diary") that marks a task for the agent.
4. **Other material (optional)**: what the user did on the computer that day (e.g. Claude Code or Codex session logs). If you can connect WeChat, Feishu/Lark or other chat history, you can also analyze the day's work and personal messages. Check compliance before reading chat history, and never copy passwords or keys into the diary.

## 2. Diary structure

Write to `diary/<date>.md`:

- **Timeline**: morning, afternoon, evening: what happened, where the user went, what was said. Quote accurately and add the audio position in parentheses.
- **An outside observer's view** (required), in four parts:
  - **What went well**: what exactly was done right, and why it was good.
  - **What could be better**: be direct, no flattery, and give a better way to do it. Correct factual or common-sense mistakes.
  - **Opportunities you may have missed**: both money and happiness, e.g. idle resources, skills or work that could earn money, emotional moments with family.
  - **Try next**: one to three small, concrete things to do right away.
- **Questions you asked in the recording**: answer each one; you may search the web, and cite sources.
- **To-dos**: action items pulled from the conversations.
- **Recording notes**: which hours were recorded, the offset between audio time and real time, and how reliable the speaker labels are.

Keep the questions and to-dos short.

## 3. Rules

- **Every observation needs evidence**: quote the words and the time; don't speculate. If only part of the day was recorded, say so; an observation is not a verdict.
- **Who said what follows the voiceprint labels, but labels can be wrong**. Write "uncertain" when unsure. Before criticizing the user, check the label on that line; don't pin someone else's words on the user.
- **Check the premise before criticizing**. Facts about the household (who picks up the kids, who is free when) and technical approaches the user has actually tried go by what is recorded in `memory/` and by the user's own results, not by general knowledge or textbook claims.
- **Advice must be doable** under the user's actual circumstances. Keep the household situation in `memory/`; the user's corrections win.
- **Privacy**: recordings, transcripts and diaries are processed only on this machine (and your own GPU host), never uploaded to any other online service, and never committed to git (`.gitignore` already excludes them). Sub-agents that read transcripts must also run locally.

## 4. Memory

Over time the diary accumulates facts about the user: the household, people who come up often, names the transcription keeps getting wrong, and the user's corrections. Keep them in a `memory/` folder in the project, one fact per file, indexed by `memory/MEMORY.md`, and reference that index from `CLAUDE.md`. `memory/` is in `.gitignore` and is never committed.

Each memory file looks like this:

```markdown
---
name: family-situation
description: One-line summary, used to decide relevance when recalling
metadata:
  type: user | feedback | project | reference
---

The fact itself. For feedback and project types, add a **Why:** line and a **How to apply:** line.
```

Before writing, check whether an existing file already covers it and update that one; delete memories that turn out to be wrong.
