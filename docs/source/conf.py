# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('../../src/neuralkg_ind'))
import doctest

project = 'NeuralKG-ind'
copyright = '2023, zjukg'
author = 'zjukg'
release = 'v1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
'sphinx.ext.autodoc',
'sphinx.ext.autosummary',
'sphinx.ext.doctest',
'sphinx.ext.intersphinx',
'sphinx.ext.mathjax',

'sphinx.ext.napoleon',
'sphinx.ext.viewcode',
'sphinx.ext.githubpages',

'sphinx.ext.todo',
'sphinx.ext.coverage',

'recommonmark',
'sphinx_markdown_tables',
]

templates_path = ['_templates']
exclude_patterns = []


doctest_default_flags = doctest.NORMALIZE_WHITESPACE
autodoc_member_order = 'bysource'
intersphinx_mapping = {'python': ('https://docs.python.org/', None)}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import sphinx_rtd_theme
#html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_static_path = ['_static']

html_css_files = ['css/custom.css']
# html_logo = './_static/logo.png'

html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "zjukg", # Username
    "github_repo": "NeuralKG_ind", # Repo name
    "github_version": "main", # Version
    "conf_py_path": "/docs/source/", # Path in the checkout to the docs root
}