# TODO

## 文档站点

- [ ] 待 zensical 原生支持 Git metadata（贡献者列表、修订日期）后，恢复页面底部的贡献者与修订日期显示。

  背景：项目原先通过 `extensions/git_info.py` 桥接 mkdocs 的 `mkdocs-git-committers-plugin-2` 和 `mkdocs-git-revision-date-localized-plugin` 实现该功能。该扩展在每个页面渲染时调用 GitHub API，无 token 时触发 rate limit，网络不稳定时阻塞构建导致页面空白，已移除。

  zensical 官方在 [插件兼容性页面](https://zensical.org/compatibility/plugins/) 列出 git-committers、git-revision-date-localized 等功能计划原生支持，但目前尚未实现（属于未来的 module system）。届时可直接在 `zensical.toml` 中配置，无需再写桥接代码。
