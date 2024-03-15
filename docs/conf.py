# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
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
    "sphinx_design",
    "sphinx_favicon",
]

# show tocs for classes and functions of modules using the autodocsumm
# package
autodoc_default_options = {'autosummary': True}

# show the code of plots that follows the command .. plot:: based on the
# package matplotlib.sphinxext.plot_directive
plot_include_source = True

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

project = 'pyfar gallery'
copyright = '2020-2024, The pyfar developers'
author = 'The pyfar developers'
release = '0.1.0'



templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# default language for highlighting in source code
highlight_language = "python3"

# intersphinx mapping
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'spharpy': ('https://spharpy.readthedocs.io/en/stable/', None)
    }

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ['css/custom.css']
html_logo = 'resources/logos/pyfar_logos_fixed_size_pyfar.png'
html_title = "pyfar"
html_favicon = '_static/favicon.ico'


# -- HTML theme options
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/layout.html

html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["navbar-icon-links"],
    "navbar_align": "content",
    "icon_links": [
        {
          "name": "GitHub",
          "url": "https://github.com/pyfar",
          "icon": "fa-brands fa-square-github",
          "type": "fontawesome",
        },
        {
            "name": "CC-BY",
            "url": "https://creativecommons.org/licenses/by/4.0/deed.de",
            "icon": "fa-brands fa-creative-commons-by",
            "type": "fontawesome",
        }
    ],

  # Configure secondary (right) side bar
  "show_toc_level": 3,                     # Show all subsections of notebooks
  "secondary_sidebar_items": ["page-toc"]  # Omit 'show source' link that that
                                           # shows notebook in json format
}

html_context = {
   "default_mode": "light"
}
# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
# texinfo_documents = [
#     (master_doc, 'pyfar',
#      u'pyfar Gallery',
#      author,
#      'pyfar',
#      'One line description of project.',
#      'Miscellaneous'),
# ]

# -- Options for nbsphinx -------------------------------------------------
nbsphinx_prolog = r"""
{% set docname = 'doc/' + env.doc2path(env.docname, base=None) %}

.. raw:: html

    <div class="admonition note">
      Open an interactive online version by clicking the badge
      <span style="white-space: nowrap;"><a href="https://mybinder.org/v2/gh/pyfar/gallery/main?filepath={{ docname|e }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a></span>
      or
      <a href="{{ env.docname.split('/')|last|e + '.ipynb' }}" class="reference download internal" download>download</a>
      the notebook.
      <script>
        if (document.location.host) {
          let nbviewer_link = document.createElement('a');
          nbviewer_link.setAttribute('href',
            'https://nbviewer.org/url' +
            (window.location.protocol == 'https:' ? 's/' : '/') +
            window.location.host +
            window.location.pathname.slice(0, -4) +
            'ipynb');
          nbviewer_link.innerHTML = 'Or view it on <em>nbviewer</em>';
          nbviewer_link.classList.add('reference');
          nbviewer_link.classList.add('external');
          document.currentScript.replaceWith(nbviewer_link, '.');
        }
      </script>
    </div>

"""

# -- manage thumbnails --------------------------------------------------------
# must be located in 'docs/_static'
nbsphinx_thumbnails = {
}


# -- pyfar specifics -----------------------------------------------------

# write shortcuts to sphinx readable format
_, shortcuts = pyfar.plot.shortcuts(show=False, report=True, layout="sphinx")
shortcuts_path = os.path.join("concepts", "resources", "plot_shortcuts.rst")
with open(shortcuts_path, "w") as f_id:
    f_id.writelines(shortcuts)
