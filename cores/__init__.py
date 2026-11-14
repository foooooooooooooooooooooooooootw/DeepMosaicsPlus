# Caches and temp files go next to the project, not on C: (see util/storage.py).
# Every entry point imports cores before it uses torch, so this runs early enough.
from util import storage as _storage
_storage.redirect_caches()
from .options import *