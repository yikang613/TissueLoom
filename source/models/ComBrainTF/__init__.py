"""ComBrainTF subpackage -- Community-aware Brain Transformer.

Ported from ``ubc-tea/Com-BrainTF`` (``source/models/COMTF/comtf.py``)
into the BrainNetworkTransformer convention. The exported symbol is
:class:`ComBrainTF`, matching the class name expected by the model
factory in ``source/models/__init__.py``.
"""

from .comtf import ComBrainTF

__all__ = ["ComBrainTF"]
