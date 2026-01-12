---
name: launch-agents
description: Launching Codex and Claude agents as bash commands with prompts via stdin
---

# Launch Agents

## Overview
Run Codex (OpenAI) and Claude (Anthropic) CLI agents from bash, passing prompts via stdin.

## Commands

Use these exact flags. Only the output destination can be changed.

### Codex
```bash
echo "Your prompt" | codex exec \
  --dangerously-bypass-approvals-and-sandbox \
  --model gpt-5.2-codex \
  -c reasoning_effort=xhigh \
  -o output.md
```

### Claude
```bash
echo "Your prompt" | claude -p \
  --dangerously-skip-permissions \
  --model opus \
  --output-format text
```

## Example

```bash
cat task.md | codex exec --dangerously-bypass-approvals-and-sandbox --model gpt-5.2-codex -o result.md

cat <<'EOF' | claude -p --dangerously-skip-permissions --model opus --output-format text
Review the changes in result.md. Reply [COMPLETE] if done.
EOF
```
