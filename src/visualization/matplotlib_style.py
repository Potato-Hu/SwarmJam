from __future__ import annotations

import matplotlib as mpl


def configure_matplotlib_for_ieee_pdf() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
        }
    )


configure_matplotlib_for_ieee_pdf()
