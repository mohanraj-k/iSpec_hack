"""Central location for match quality thresholds used across backend and UI.

Modify these values to adjust matching logic and automatically reflect in
front-end legends (via `app.py`). All values are expressed as decimal
fractions (e.g. 0.6 == 60 %).
"""

EXCELLENT = 0.80  # ≥ 80 %
GOOD      = 0.60  # ≥ 60 %
MODERATE  = 0.40  # ≥ 40 %

# Weak is anything below MODERATE, no explicit threshold required
# Dict exported for convenience when passing to templates
THRESHOLDS = {
    "excellent": EXCELLENT,
    "good": GOOD,
    "moderate": MODERATE,
    "weak": 0.0,
}
