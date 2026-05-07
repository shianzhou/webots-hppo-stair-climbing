# webots-hppo-stair-climbing
Multi-Agent HPPO
HPPO-based reinforcement learning for stair climbing in Webots.


注意事项与单智能体一样，有问题给我说

## 项目代码说明网站

线上网站入口：

[打开项目代码说明网站](https://shianzhou.github.io/webots-hppo-stair-climbing/)

[文档首页源码](docs_site_staging/docs/index.md)

本地预览：

```powershell
conda activate rl_env
python -m mkdocs serve -f docs_site_staging\mkdocs.yml
```

浏览器打开：

```text
http://127.0.0.1:8000/
```

文档源文件在 [docs_site_staging/docs](docs_site_staging/docs)，发布清单在 [docs_site_staging/publish.yml](docs_site_staging/publish.yml)。当前只同步和项目代码说明相关的 Obsidian 笔记。

注意：`docs_site_staging/site` 是本地构建产物，不提交到 GitHub。GitHub Pages 会通过 Actions 自动构建并发布网站。
