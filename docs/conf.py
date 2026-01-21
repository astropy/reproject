# Licensed under a 3-clause BSD style license - see LICENSE.rst

import datetime
import sys
from importlib import metadata

try:
    from sphinx_astropy.conf.v2 import *  # noqa
except ImportError:
    print("ERROR: the documentation requires the sphinx-astropy package to be installed")
    sys.exit(1)

# -- General configuration ----------------------------------------------------

# By default, highlight as Python 3.
highlight_language = "python3"

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.1'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns.append("_templates")

# This is added to the end of RST files - a good place to put substitutions to
# be used globally.
rst_epilog += """
"""

# -- Project information ------------------------------------------------------

package_info = metadata.metadata("reproject")

# This does not *have* to match the package name, but typically does
project = package_info["Name"]
author = package_info["Author"]
copyright = "{}, {}".format(datetime.datetime.now().year, package_info["Author"])

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

# The short X.Y version.
version = package_info["Version"].split("-", 1)[0]
# The full version, including alpha/beta/rc tags.
release = package_info["Version"]


# -- Options for HTML output ---------------------------------------------------

html_theme = "astropy-unified"

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = f"{project} v{release}"

# Output file base name for HTML help builder.
htmlhelp_basename = project + "doc"


# -- Options for LaTeX output --------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [("index", project + ".tex", project + " Documentation", author, "manual")]


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [("index", project.lower(), project + " Documentation", [author], 1)]


## -- Options for the edit_on_github extension ----------------------------------------

nitpicky = True

plot_rcparams = {}
plot_rcparams["figure.figsize"] = (8, 6)
plot_rcparams["savefig.facecolor"] = "none"
plot_rcparams["savefig.bbox"] = "tight"
plot_rcparams["axes.labelsize"] = "large"
plot_rcparams["figure.subplot.hspace"] = 0.5

plot_apply_rcparams = True
plot_html_show_source_link = False
plot_formats = ["png", "svg", "pdf"]
