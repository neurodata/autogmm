import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

nb_in, nb_out = sys.argv[1], sys.argv[2]
with open(nb_in) as f:
    nb = nbformat.read(f, as_version=4)
ep = ExecutePreprocessor(timeout=3600, kernel_name="python3")
ep.preprocess(nb, {})
with open(nb_out, "w") as f:
    nbformat.write(nb, f)
