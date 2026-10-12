"""``python -m mdatk`` entry point (same as the ``mdatk`` command)."""

import sys

from .cli import main

sys.exit(main())
