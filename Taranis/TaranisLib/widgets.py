"""Small Qt helpers shared by the Taranis modules."""

import qt

MIN_CONTENTS_CHARACTERS = 8   # a combo box shows at least this many characters, but never asks for more


def allowNarrowPanel(root):
    """Long volume / segment names must not widen the module panel: every combo box below root (also the ones
    inside node and segment selectors) sizes itself from a few characters instead of its longest item, and long
    items are elided. Safe to call repeatedly (e.g. after rows were added)."""
    if root is None:
        return
    for combo in root.findChildren("QComboBox"):
        try:
            combo.setSizeAdjustPolicy(qt.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(MIN_CONTENTS_CHARACTERS)
            policy = combo.sizePolicy
            policy.setHorizontalPolicy(qt.QSizePolicy.Expanding)
            combo.setSizePolicy(policy)
            # Qt caches the minimum size hint from the longest item; an explicit minimum width replaces it
            if combo.minimumWidth == 0:
                combo.setMinimumWidth(60)
            # node selectors (ctkComboBox) keep asking for their current item's full width whatever the policy;
            # containers that may not shrink below their preferred width (collapsible sections) would pass that on
            parent = combo.parentWidget()
            while parent is not None:
                policy = parent.sizePolicy
                if policy.horizontalPolicy() == qt.QSizePolicy.Minimum:
                    policy.setHorizontalPolicy(qt.QSizePolicy.Preferred)
                    parent.setSizePolicy(policy)
                if parent is root:
                    break
                parent = parent.parentWidget()
        except Exception:
            pass
