# Security policy

## Secrets

- This repository must never contain real credentials. `.env` and `.env.*` (except `.env.example`), private keys, `kaggle.json`,
  `credentials*.json` and `.dvc/config.local` are git-ignored.
- `.env.example` lists variable **names** only. Supply your own keys through `.env` or your shell; never through issues, pull requests or commit messages.
- Secret settings are hidden from `repr()`, logs and API responses; `prodml-doctor` reports them only as "set/unset".
- If a key is ever committed, revoke it at the provider first, then remove it from history.

## Reporting a vulnerability

Open a private security advisory on the GitHub repository (Security tab -> "Report a vulnerability") or contact the maintainer directly.
Please do not file public issues containing exploit details or credentials.

## Scope and limits

The service gives informational retrieval over statute text, not legal advice. The Grafana admin password in `docker/docker-compose.yml` is the default `admin`:
the compose stack is for local demonstration and must not be exposed to a network.
