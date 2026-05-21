"""
Markdown extension that bridges MkDocs git plugins into zensical.

Calls the actual mkdocs-git-committers-plugin-2 and
mkdocs-git-revision-date-localized-plugin during markdown preprocessing,
injecting their output into page.meta so zensical's template can render it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from markdown import Extension
from markdown.preprocessors import Preprocessor

if TYPE_CHECKING:
    from markdown import Markdown

log = logging.getLogger("zensical.extensions.git_info")

DOCS_DIR = os.path.join(os.getcwd(), "docs")


class _MockFile:
    def __init__(self, src_path: str):
        self.src_path = src_path
        self.abs_src_path = str(Path(DOCS_DIR) / src_path)
        self.generated_by = None
        self.locale = None

    def is_documentation_page(self) -> bool:
        return True


class _MockPage:
    def __init__(self, file: _MockFile, meta: dict):
        self.file = file
        self.meta = meta


class _MockConfig(dict):
    def __init__(self, data: dict):
        super().__init__(data)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


_committers_plugin = None
_date_plugin = None
_initialized = False


def _init_plugins(repository: str, branch: str, enable_creation_date: bool):
    global _committers_plugin, _date_plugin, _initialized
    if _initialized:
        return
    _initialized = True

    mock_config = _MockConfig({
        "config_file_path": "zensical.toml",
        "docs_dir": DOCS_DIR,
        "plugins": {},
        "theme": {},
        "extra_javascript": [],
        "extra_css": [],
    })

    try:
        from mkdocs_git_committers_plugin_2.plugin import GitCommittersPlugin
        _committers_plugin = GitCommittersPlugin()
        _committers_plugin.config = {
            "enterprise_hostname": "",
            "gitlab_hostname": "",
            "repository": repository,
            "gitlab_repository": 0,
            "api_version": None,
            "branch": branch,
            "docs_path": "docs/",
            "enabled": True,
            "cache_dir": ".cache/plugin/git-committers",
            "exclude": [],
            "exclude_committers": [],
            "token": "",
        }
        _committers_plugin.on_config(mock_config)
        _committers_plugin.on_pre_build(mock_config)
        log.info("git-committers plugin bridge initialized")
    except Exception as e:
        log.warning(f"Failed to initialize git-committers plugin: {e}")
        _committers_plugin = None

    try:
        from mkdocs_git_revision_date_localized_plugin.plugin import (
            GitRevisionDateLocalizedPlugin,
        )
        _date_plugin = GitRevisionDateLocalizedPlugin()
        _date_plugin.config = {
            "fallback_to_build_date": False,
            "locale": None,
            "type": "date",
            "custom_format": "%d. %B %Y",
            "timezone": "UTC",
            "exclude": [],
            "enable_creation_date": enable_creation_date,
            "enabled": True,
            "strict": True,
            "enable_git_follow": True,
            "ignored_commits_file": None,
            "enable_parallel_processing": False,
        }
        _date_plugin.on_config(mock_config)
        log.info("git-revision-date-localized plugin bridge initialized")
    except Exception as e:
        log.warning(f"Failed to initialize git-revision-date-localized plugin: {e}")
        _date_plugin = None


class GitInfoPreprocessor(Preprocessor):
    name = "git_info"

    def __init__(self, md: Markdown, repository: str, branch: str, enable_creation_date: bool):
        super().__init__(md)
        self.repository = repository
        self.branch = branch
        self.enable_creation_date = enable_creation_date

    def run(self, lines: list[str]) -> list[str]:
        from zensical.extensions.context import ContextPreprocessor

        ctx = ContextPreprocessor.from_markdown(self.md)
        if not ctx:
            return lines

        _init_plugins(self.repository, self.branch, self.enable_creation_date)

        page = ctx.page
        mock_file = _MockFile(page.path)
        mock_page = _MockPage(mock_file, page.meta)

        if _date_plugin:
            try:
                _date_plugin.on_page_markdown(
                    "\n".join(lines), mock_page, {}, None
                )
                page.meta.update(mock_page.meta)
            except Exception as e:
                log.warning(f"git-revision-date-localized failed for {page.path}: {e}")

        if _committers_plugin:
            try:
                context: dict[str, Any] = {}
                _committers_plugin.on_page_context(
                    context, mock_page, {}, None
                )
                if context.get("committers"):
                    page.meta["committers"] = context["committers"]
            except Exception as e:
                log.warning(f"git-committers failed for {page.path}: {e}")

        return lines


class GitInfoExtension(Extension):
    name = "extensions.git_info"

    def __init__(self, **kwargs: Any):
        self.config = {
            "repository": ["", "GitHub repository (owner/repo)"],
            "branch": ["main", "Branch name"],
            "enable_creation_date": [True, "Enable creation date"],
        }
        super().__init__(**kwargs)

    def extendMarkdown(self, md: Markdown) -> None:
        md.registerExtension(self)
        preprocessor = GitInfoPreprocessor(
            md=md,
            repository=self.getConfig("repository"),
            branch=self.getConfig("branch"),
            enable_creation_date=self.getConfig("enable_creation_date"),
        )
        md.preprocessors.register(preprocessor, preprocessor.name, 1)


def makeExtension(**kwargs: Any) -> GitInfoExtension:
    return GitInfoExtension(**kwargs)
