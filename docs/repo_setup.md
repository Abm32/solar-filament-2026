# Repository Setup

## GitHub

The local GitHub CLI account exists but the token is invalid:

```text
gh auth status -> token invalid for account Abm32
```

After re-authentication:

```bash
gh auth login -h github.com
gh repo create solar-filament-2026 --public --source=. --remote=origin --push
```

## Commit Rhythm

Commit after each useful milestone:

- scaffold and research notes;
- data download and EDA;
- baseline submission;
- first trainable model;
- post-processing improvements;
- final report assets.

