# Data layout

| Path | Contents |
| --- | --- |
| `init/` | Synthetic smoke package (no licensed data) |
| `external/` | Local LUMPI / Synthehicle downloads (gitignored) |
| `work/` | Imported and remerged SQLite files (gitignored) |

Create the smoke package with:

```text
python scripts/init_smoke_data.py
```

Do not commit licensed videos, GT world coordinates, or large work
databases to the public artifact repository.
