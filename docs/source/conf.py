import os
import sys

# inject path to maniskill package to enable autodoc/autoapi to find packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../mani_skill"))
import mani_skill
__version__ = mani_skill.__version__

project = "ManiSkill"
copyright = "2024, ManiSkill Contributors"
author = "ManiSkill Contributors"
release = __version__
version = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx_subfigure",
    "sphinxcontrib.video",
    "sphinx_togglebutton",
    "sphinx_design",
    "autoapi.extension",
]

# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html
myst_enable_extensions = ["colon_fence", "dollarmath"]
# https://github.com/executablebooks/MyST-Parser/issues/519#issuecomment-1037239655
myst_heading_anchors = 4

templates_path = ["_templates"]

html_theme = "pydata_sphinx_theme"
html_logo = "_static/logo_black.svg"
html_favicon = "_static/favicon.svg"

json_url = "_static/version_switcher.json"
version_match = os.environ.get("READTHEDOCS_VERSION")
if version_match is None:
    version_match = "v" + __version__
html_theme_options = {
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/haosulab/ManiSkill",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Website",
            "url": "https://maniskill.ai",
            "icon": "fa-solid fa-globe",
        }
    ],
    "external_links": [
        {"name": "Changelog", "url": "https://github.com/haosulab/ManiSkill/releases"},
    ],
    "logo": {
        "image_dark": "_static/logo_white.svg",
    },
    "navbar_center": ["version-switcher", "navbar-nav"],
    "show_version_warning_banner": False,
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
}
html_context = {
    "display_github": True,
    "github_user": "haosulab",
    "github_repo": "ManiSkill",
    "github_version": "main",
    "conf_py_path": "/source/",
    "doc_path": "docs/source"
}
html_css_files = [
    'css/custom.css',
]
html_static_path = ['_static']

# autodoc configs
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "groupwise"

# autoapi configs
autoapi_type = "python"
autoapi_dirs = ["../../mani_skill/"]
autoapi_options =  ['members', 'undoc-members', 'private-members', 'show-inheritance', 'show-module-summary', 'special-members', 'imported-members', ]
# there's quite a few files that do not need to be documented because they just contain e.g. example scripts or
# some very specific files for specific objects (e.g. some scene builders), or are just internally used functions that
# are not meant to be used by the user
autoapi_ignore = [
    "*/mani_skill/utils/scene_builder/*.py",
    "*/mani_skill/agents/robots/*.py",
    "*/mani_skill/examples/*.py",
    "*/mani_skill/render/*.py",
    # depth_camera is outdated and needs to be upgraded
    "*/mani_skill/sensors/depth_camera.py",
]
autoapi_keep_files = True
autoapi_root = "api"
autoapi_member_order = "groupwise"

# Intersphinx mapping to enable referencing other package docs
intersphinx_mapping = {'gymnasium': ('https://gymnasium.farama.org/', None)}
