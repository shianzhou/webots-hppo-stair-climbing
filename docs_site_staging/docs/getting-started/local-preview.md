# Local Preview

在 Windows PowerShell 中运行：

```powershell
cd D:\rl-research-docs
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python scripts\sync_obsidian.py
mkdocs serve
```

然后打开终端显示的本地地址，通常是：

```text
http://127.0.0.1:8000/
```

构建检查：

```powershell
mkdocs build --strict
```
