"""DHGFormer subpackage -- Dynamic Hierarchical Graph Transformer.

Ported from the stand-alone DHGFormer repo (MICCAI 2025) into the
BrainNetworkTransformer convention. The single exported symbol is
:class:`DHGFormer`, matching the class name expected by the model
factory in ``source/models/__init__.py``.
"""

from .dhgformer import DHGFormer

__all__ = ["DHGFormer"]
