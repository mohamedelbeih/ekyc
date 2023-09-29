"""
    Provide rapidoc ui for swagger documentations.
"""
from flask import Blueprint
from flask import  render_template_string
from flask import  url_for

blueprint = Blueprint('rapidocs', __name__)

TEMPLATE = """
<!doctype html> <!-- Important: must specify -->
<html>
  <head>
    <meta charset="utf-8"> <!-- Important: rapi-doc uses utf8 characters -->
    <script type="module" src="https://unpkg.com/rapidoc/dist/rapidoc-min.js"></script>
  </head>
  <body>
    <rapi-doc spec-url = "{{ doc_file_url }}"> </rapi-doc>
  </body>
</html>
"""


@blueprint.route("/docs", methods=["GET"])
def all_docs_view():
    '''
      This function view all docs in the system into rapidocs.
    '''
    return render_template_string(
      TEMPLATE,
      doc_file_url=url_for(
        "static",
        filename="docs.yml"
      )
    )
