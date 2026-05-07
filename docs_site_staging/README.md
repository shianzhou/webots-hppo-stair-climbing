# webots-hppo-stair-climbing

MkDocs + Material documentation site for Webots HPPO stair-climbing research notes.

## Local preview

```powershell
cd D:\rl-research-docs
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python scripts\sync_obsidian.py
mkdocs serve
```

## Select published notes

Edit `publish.yml`, then run:

```powershell
python scripts\sync_obsidian.py --dry-run
python scripts\sync_obsidian.py
```

Only files selected in `publish.yml` are copied from Obsidian into `docs/notes`.
