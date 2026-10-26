# faiss must load its bundled libomp before torch loads its own copy; the reverse order
# segfaults on macOS. Test modules import torch, so load faiss here first.
import heads.knn  # noqa: F401
