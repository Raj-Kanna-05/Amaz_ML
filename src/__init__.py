"""
Root src module forwarding to code.business_entity_resolution.src
"""
import os
import sys

_SRC_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "code",
    "business_entity_resolution",
    "src"
)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
