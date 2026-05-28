"""LRBGT subpackage -- Local Random-walk Brain Graph Transformer.

Ported from the stand-alone ALTER repo (``yushuowiki/ALTER``,
``alter/models/LRBGT/lrbgt.py``) into the BrainNetworkTransformer
convention. The original file defined a class named
``BrainNetworkTransformer`` which collides with BNT's existing class of
the same name, so we rename the public symbol to :class:`LRBGT` here.

The only exported symbol is :class:`LRBGT`, matching the class name
expected by the model factory in ``source/models/__init__.py``.
"""

from .lrbgt import LRBGT

__all__ = ["LRBGT"]
