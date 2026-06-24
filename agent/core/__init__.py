"""agent/core -- the v024-aligned self-solve loop (seed of the real agent).

Grown empirically by the thinnest vertical slice (ls20 L1 self-solve). Reuses the
general primitives from agent/ (segment, controllable) by import; never imports the
belief-era modules and never imports tools/ (the test environment lives there).
"""
