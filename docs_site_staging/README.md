# webots-hppo-stair-climbing

MkDocs + Material documentation site for Webots HPPO stair-climbing project code notes.

[打开项目代码说明网站](https://shianzhou.github.io/webots-hppo-stair-climbing/)

[文档首页源码](docs/index.md)

`site/` is a local build output and is intentionally ignored by Git. GitHub Pages is published by the root workflow `.github/workflows/deploy.yml`.

## Local preview

```powershell
cd E:\project_MultiAgent_h_change
conda activate rl_env
python docs_site_staging\scripts\sync_obsidian.py
python -m mkdocs serve -f docs_site_staging\mkdocs.yml
```

## Select published notes

Edit `publish.yml`, then run:

```powershell
python docs_site_staging\scripts\sync_obsidian.py --dry-run
python docs_site_staging\scripts\sync_obsidian.py
```

Only files selected in `publish.yml` are copied from Obsidian into `docs/notes`.
