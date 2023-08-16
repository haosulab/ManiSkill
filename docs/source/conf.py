# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ManiSkill2"
copyright = "2023, ManiSkill2 Contributors"
author = "ManiSkill2 Contributors"
release = "0.5.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx_subfigure",
]

# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html
myst_enable_extensions = ["colon_fence", "dollarmath"]
# https://github.com/executablebooks/MyST-Parser/issues/519#issuecomment-1037239655
myst_heading_anchors = 4

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]

# replace "view page source" with "edit on github" in Read The Docs theme
#  * https://github.com/readthedocs/sphinx_rtd_theme/issues/529
html_context = {
    "display_github": True,
    "github_user": "haosulab",
    "github_repo": "ManiSkill2",
    "github_version": "main/docs/source/",
}
