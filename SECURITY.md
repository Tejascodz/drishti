# Security policy

## Reporting a vulnerability

Report vulnerabilities privately through GitHub: **Security → Report a vulnerability**
on this repository. Do not open a public issue. Include the affected version or commit,
steps to reproduce and the impact you expect. You should get a first reply within a week.

## Supported versions

Only the latest commit on `main` receives fixes.

## Deployment guidance

* **REST API.** Set `DRISHTI_API_KEY` whenever `drishti serve` or the Docker image is
  reachable from a network. Without it every `/v1` endpoint (recognition, stored events,
  watchlist) is open. Put the API behind TLS (a reverse proxy) outside a trusted network.
* **Model downloads.** Hub models are verified against pinned SHA-256 digests in
  `drishti/hub.py`; a mismatching file is discarded. Your own ONNX files are loaded as
  given, so keep them in a location only you can write to.
* **Plate data.** Plate numbers, crops and event databases are personal data in many
  jurisdictions. Use `privacy.hash_plates` (HMAC with a key from `DRISHTI_HMAC_KEY`),
  `privacy.blur_non_matches` and `privacy.retention_days`, and keep datasets out of
  version control (the `.gitignore` excludes them).
* **Webhooks.** `alerts.webhook_url` receives plate events on watchlist hits; use HTTPS.
