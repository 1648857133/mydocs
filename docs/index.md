# 欢迎来到mkdoc

完整文档请访问 [mkdocs.org](https://www.mkdocs.org)。

本地预览网址：[http://127.0.0.1:8000/](http://127.0.0.1:8000/)

## 常用命令

- `mkdocs new [目录名]` - 创建一个新项目。
- `mkdocs serve` - 启动实时预览文档服务器。
- `mkdocs build` - 构建文档网站。
- `mkdocs -h` - 显示帮助信息并退出。

## 项目结构

    mkdocs.yml    # 配置文件
    docs/
        index.md  # 文档首页
        ...       # 其他 Markdown 页面、图片及其他文件

## 配置页面导航
```yml
nav:
  - Home: index.md
  - User Guide:
    - Writing your docs: writing-your-docs.md
    - Styling your docs: styling-your-docs.md
  - About:
    - License: license.md
    - Release Notes: release-notes.md
```
