"""``python -m jaxaero.case [--write]``: print (or write) docs/case_reference.md."""

import sys

from .schema import REFERENCE, reference

if sys.argv[1:] == ["--write"]:
    REFERENCE.parent.mkdir(exist_ok=True)
    REFERENCE.write_text(reference())
    print(REFERENCE)
else:
    sys.stdout.write(reference())
