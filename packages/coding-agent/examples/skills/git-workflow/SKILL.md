---
name: git-workflow
description: Steps for keeping local work tidy while collaborating.
---

When working in a git repo:

1. Before editing, run `git status` and note the current branch.
2. Create focused branches; avoid mixing unrelated changes.
3. Keep commits small: stage related files together and write descriptive messages.
4. Rebase frequently on main to reduce conflicts; resolve with `git status` + `git rebase --continue`.
5. Run tests or linters that match the code touched; record failures in the session.
6. Before push/PR, `git diff` to self-review and ensure no secrets are present.
