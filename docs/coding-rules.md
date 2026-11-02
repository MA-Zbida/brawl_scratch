# Mandatory Implementation Rules (Derived from CLAUDE.md)

These rules are mandatory for all coding tasks in this repository.

## 1) Think before coding
- State assumptions explicitly before implementing when ambiguity exists.
- If multiple interpretations exist, present options instead of silently picking one.
- If unclear, stop and ask a focused clarification question.

## 2) Simplicity first
- Implement only what was requested.
- No speculative abstractions or future-proofing not required by the task.
- Prefer the smallest correct change set.

## 3) Surgical changes only
- Touch only files and lines needed for the current requirement.
- Do not refactor unrelated code.
- Preserve existing project style and conventions.
- Remove only unused code introduced by your own changes.

## 4) Goal-driven execution
- Define concrete verification goals before coding.
- For each step, include a verification check (tests/build/behavior check).
- Loop until checks pass or a real blocker is identified.
