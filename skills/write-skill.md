---
name: write-skill
description: Help user create a new skill - interview them about the task, verify assumptions, ask clarifying questions, then write the skill file
---

# Write Skill

You are helping the user create a new skill. They've identified a task they repeat often and want to codify it.

## Your Job

1. **Understand the task** - Ask what they're trying to accomplish
2. **Verify assumptions** - Challenge vague or incomplete descriptions
3. **Fill gaps** - Ask targeted follow-up questions
4. **Write the skill** - Produce a complete, working skill file
5. **Update AGENTS.md** - Add the skill to the table

## Interview Process

Start by asking the user to describe the task. Then ask follow-up questions to fill gaps:

**Essential questions (ask if not covered):**
- What triggers this task? (keywords, file types, scenarios)
- What are the concrete steps?
- What commands or code are involved?
- What mistakes should be avoided?

**Verify assumptions:**
- "You mentioned X—does that mean Y, or something else?"
- "What happens if Z fails?"
- "Are there variations of this task, or is it always the same?"

**Keep it efficient.** Don't ask questions the user already answered. Batch related questions together.

## Writing the Skill

Once you have enough context, write the skill file to `skills/<name>.md`.

### Structure

```markdown
---
name: <lowercase-with-hyphens>
description: <specific enough for you to know when to trigger it>
---

# <Skill Name>

## Overview
<2-3 sentences: what, when, why>

## Workflow
<Concrete steps with actual commands/code>

## Example
<One realistic example>

## Common Mistakes
<Only if there are genuine pitfalls>
```

### Principles

- **Concrete over conceptual**: Commands and code, not explanations of commands
- **Lean**: Only include what's needed. No filler.
- **Tested**: Every example must be executed and verified before inclusion. No untested examples.
- **Current**: Use project conventions (`uv run`, `./ivytest.sh`, etc.)

### Verifying Examples

**Before finalizing the skill, you must test every code example and command.**

1. Execute each command/snippet in the skill
2. If it fails, fix it and re-test
3. Iterate until all examples work
4. Only then include them in the skill

This is non-negotiable. A skill with broken examples is worse than no skill—it wastes time and erodes trust.

### Description Quality

The description determines when you'll trigger this skill in the future. Make it specific:

- Good: `Replaying divergences from fuzzer output - finding root cause, debugging workflow`
- Bad: `Helper for debugging` (too vague)
- Bad: `Everything about the fuzzer` (too broad)

## After Writing

1. **Test all examples** - Execute every command and code snippet. Fix any that fail.
2. Save the skill to `skills/<name>.md`
3. Add an entry to the Skills table in `AGENTS.md`
4. Confirm with the user and ask if anything is missing

## Example Interaction

**User:** I keep having to debug divergences from the fuzzer. Want to make a skill for it.

**Claude:** What does debugging a divergence involve? Walk me through a typical case.

**User:** I get a divergence file, replay it to reproduce, then narrow down which operation caused the diff...

**Claude:** A few follow-ups:
- What format is the divergence file?
- What command do you use to replay?
- How do you narrow down the operation—binary search, or something else?

**User:** [answers]

**Claude:** Got it. Let me write the skill. [writes file]

Does this capture the workflow? Anything missing about edge cases or common mistakes?
