"""Taranis shared library: case model, workflow status engine and workflow toolbar.

Used by the Taranis hub module and, over time, by the step modules (EasyReg, LSFcalc, dosimetry).
"""

import logging

_SUBMODULES = ["roles", "workflow", "doseguard", "visibility", "case", "controller", "segmentops", "ai", "segtools", "views", "lsf", "timing", "memory", "toolbar"]

# Instances replaced by a developer reload are kept alive: if Python frees their Qt widgets before Qt has
# processed the deferred deletes (deleteLater), Slicer crashes.
_retired = []


def startup():
    """Create the workflow controller and toolbar once Slicer has a main window (safe to call repeatedly)."""
    import slicer
    if slicer.util.mainWindow() is None:
        return
    from .controller import WorkflowController
    from .toolbar import WorkflowToolbar
    WorkflowController.instance()
    WorkflowToolbar.instance()
    try:
        from .views import installLayoutRegistration
        installLayoutRegistration()   # scenes saved in the segmentation layout (single or dual monitor)
    except Exception as e:
        logging.warning(f"Taranis: could not register the segmentation layouts: {e}")
    try:
        from .segtools import guardAutoCompleteEffect
        guardAutoCompleteEffect()   # Slicer's Fill between slices error before Initialize
    except Exception as e:
        logging.warning(f"Taranis: could not guard the auto-complete segment editor effects: {e}")


def shutdown():
    from .controller import WorkflowController
    from .toolbar import WorkflowToolbar
    _retired.extend(instance for instance in (WorkflowToolbar._instance, WorkflowController._instance)
                    if instance is not None)
    WorkflowToolbar.destroyInstance()
    WorkflowController.shutdownInstance()


def reloadLibrary():
    """Developer reload: remove the toolbar and controller, reload all submodules, start again."""
    import importlib
    import sys
    try:
        shutdown()
    except Exception as e:
        logging.warning(f"Taranis: shutdown before reload failed: {e}")
    for name in _SUBMODULES:
        module = sys.modules.get(f"{__name__}.{name}")
        if module is not None:
            importlib.reload(module)
    startup()
