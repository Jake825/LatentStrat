# Commit Message Template

Subject:

```text
Imperative summary under 72 characters
```

Body:

```text
Why:
- Reason for the change.

What:
- Main behavior, docs, or skill updates.

Validation:
- Command or check that was run.
```

## Example

```text
Add FRC time-aware analysis skill

Why:
- FRC analytics need explicit known_as_of guidance outside EPA-specific docs.

What:
- Added repo-scoped skill guidance and references.
- Linked scouting and Statbotics skills to the timing guidance.

Validation:
- quick_validate.py passed for the new skill.
- git diff --check passed.
```
