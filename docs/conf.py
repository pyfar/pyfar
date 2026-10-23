# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import urllib3
import shutil
import numpy as np
sys.path.insert(0, os.path.abspath('..'))

import pyfar  # noqa

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'autodocsumm',
    'sphinx_design',
    'sphinx_favicon',
    'sphinx_reredirects',
    'sphinx_mdinclude',
    'sphinx_copybutton',
]

# show tocs for classes and functions of modules using the autodocsumm
# package
autodoc_default_options = {'autosummary': True}

# show the code of plots that follows the command .. plot:: based on the
# package matplotlib.sphinxext.plot_directive
plot_include_source = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'pyfar'
copyright = "2020, The pyfar developers"
author = "The pyfar developers"

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
version = pyfar.__version__
# The full version, including alpha/beta/rc tags.
release = pyfar.__version__

# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use (Not defining
# uses the default style of the html_theme).
# pygments_style = 'sphinx'

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# default language for highlighting in source code
highlight_language = "python3"

# intersphinx mapping
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'gallery': ('https://pyfar-gallery.readthedocs.io/en/latest/', None),
    'soundfile': ('https://python-soundfile.readthedocs.io/en/latest/', None),
    'spharpy': ('https://spharpy.readthedocs.io/en/stable/', None),
    }

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ['css/custom.css']
html_js_files = ['js/custom.js']
html_logo = 'resources/logos/pyfar_logos_fixed_size_pyfar.png'
html_title = "pyfar"
html_favicon = '_static/favicon.ico'

# -- HTML theme options
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/layout.html
html_sidebars = {
  "pyfar": []
}

html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "navbar_align": "content",
    "header_links_before_dropdown": None,  # will be automatically set later based on headers.rst
    "header_dropdown_text": "Packages",  # Change dropdown name from "More" to "Packages"
    "icon_links": [
        {
          "name": "GitHub",
          "url": "https://github.com/pyfar",
          "icon": "fa-brands fa-square-github",
          "type": "fontawesome",
        },
    ],
    # Configure secondary (right) side bar
    "show_toc_level": 3,  # Show all subsections of notebooks
    "secondary_sidebar_items": ["page-toc"],  # Omit 'show source' link that that shows notebook in json format
    "navigation_with_keys": True,
    # Configure navigation depth for section navigation
    "navigation_depth": 2,
}

html_context = {
   "default_mode": "light"
}

# redirect index to pyfar.html
redirects = {
     "index": "pyfar.html"
}

# -- download navbar and style files from gallery -----------------------------
branch = 'main'
link = f'https://github.com/pyfar/gallery/raw/{branch}/docs/'
folders_in = [
    '_static/css/custom.css',
    '_static/js/custom.js',
    '_static/favicon.ico',
    '_static/header.rst',
    'resources/logos/pyfar_logos_fixed_size_pyfar.png',
    ]

def download_files_from_gallery(link, folders_in):
    c = urllib3.PoolManager()
    for file in folders_in:
        url = link + file
        filename = file
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with c.request('GET', url, preload_content=False) as res:
            if res.status == 200:
                with open(filename, 'wb') as out_file:
                    shutil.copyfileobj(res, out_file)

download_files_from_gallery(link, folders_in)
# if logo does not exist, use pyfar logo
if not os.path.exists(html_logo):
    download_files_from_gallery(
        link, ['resources/logos/pyfar_logos_fixed_size_pyfar.png'])
    shutil.copyfile(
        'resources/logos/pyfar_logos_fixed_size_pyfar.png', html_logo)


# -- modify downloaded header file from the gallery to   ----------------------
# -- aline with the local toctree ---------------------------------------------

# read the header file from the gallery
with open("_static/header.rst", "rt") as fin:
    lines = [line for line in fin]

# replace readthedocs link with internal link to this documentation
lines_mod = [
    line.replace(f'https://{project}.readthedocs.io', project) for line in lines]

# if not found, add this documentation link to the end of the list, so that
# it is in the doc tree
contains_project = any(project in line for line in lines_mod)
if not contains_project:
    lines_mod.append(f'   {project} <{project}>\n')

# write the modified header file
# to the doc\header.rst folder, so that it can be used in the documentation
with open("header.rst", "wt") as fout:
    fout.writelines(lines_mod)


# -- find position of pyfar package in toctree --------------------------------
# -- this is required to define the dropdown of Packages in the header --------

# find line where pyfar package is mentioned, to determine the start of 
# the packages list in the header
n_line_pyfar = 0
for i, line in enumerate(lines):
    if 'https://pyfar.readthedocs.io' in line:
        n_line_pyfar = i
        break

# the first 4 lines of the header file are defining the settings and a empty
# line of the toctree, therefore we need to subtract 4 from the line number
# of the pyfar package to get the correct position in the toctree
n_toctree_pyfar = n_line_pyfar - 4

if n_toctree_pyfar < 1:	
    raise ValueError(
        "Could not find the line where pyfar package is mentioned. "
        "Please check the header.rst file in the gallery."
    )

# set dropdown header at pyfar appearance, so that pyfar is the first item in
# the dropdown called Packages
html_theme_options['header_links_before_dropdown'] = n_toctree_pyfar
