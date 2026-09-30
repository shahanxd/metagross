"""METAGROSS onboard autonomy stack.

Entry points: :class:`metagross.autonomy.node.AutonomyStack` (in-process) and
:func:`metagross.autonomy.process.autonomy_main` (separate OS process behind the ipc pipe).
Kept import-free so that importing a sub-package (e.g. perception) stays cheap.
Nothing under this package may import ``metagross.sim``, ``metagross.eval`` or
``metagross.train`` (checked by ``tests/test_gt_separation.py``).
"""
