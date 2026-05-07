# Publishing Workflow

## 选择发布内容

编辑 `publish.yml`：

```yaml
entries:
  - source: "项目重构"
    dest: "project"
    enabled: true
```

- `source` 是 Obsidian 源目录中的相对路径。
- `dest` 是发布到 `docs/notes` 下的位置。
- `enabled: true` 表示发布。
- `enabled: false` 表示撤回。

## 预览同步

```powershell
python scripts\sync_obsidian.py --dry-run
```

## 同步发布副本

```powershell
python scripts\sync_obsidian.py
```

脚本会重建 `docs/notes`。这意味着从 `publish.yml` 移除或禁用的内容，下次同步后会从网站副本中消失。

## GitHub Pages 配置

1. 在 GitHub 创建仓库：`webots-hppo-stair-climbing`。
2. 把本地仓库推送到 GitHub。
3. 打开仓库 `Settings` -> `Pages`。
4. `Build and deployment` 的 `Source` 选择 `GitHub Actions`。
5. 推送到 `main` 分支后，`.github/workflows/deploy.yml` 会自动构建并部署。
