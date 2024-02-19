from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from pdfrw import PdfDict, PdfReader
from pdfrw.buildxobj import pagexobj
from pdfrw.toreportlab import makerl
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
)

from neurodesign import Optimisation

plt.switch_backend("agg")


def make_report(population: Optimisation, outfile: str | Path = "NeuroDesign.pdf"):
    """Create a report of a finished design optimisation."""
    if not isinstance(population.cov, np.ndarray):
        population.evaluate()

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(
        str(outfile),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=18,
    )

    story = []
    logofile = Path(__file__).parent / "media" / "NeuroDes.png"

    im = Image(logofile, 1 * inch, 1.25 * inch)
    story.append(im)
    story.append(Spacer(1, 12))

    title = "NeuroDesign: optimalisation report"
    story.append(Paragraph(title, styles["title"]))
    story.append(Spacer(1, 12))

    formatted_time = time.ctime()
    ptext = f"Document created: {formatted_time}"
    story.append(Paragraph(ptext, styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Correlation between designs", styles["Heading2"]))
    story.append(Spacer(1, 12))

    corr = f"""During the optimisation, the designs are mixed
with each other to find better combinations.
As such, the designs can look very similar.
Actually, the genetic algorithm uses natural selection as a basis,
and as such, the designs can be clustered in families.
This is the covariance matrix between the final {population.G} designs"""
    story.append(Paragraph(corr, styles["Normal"]))

    story.append(PageBreak())

    story.append(Paragraph("Selected designs", styles["Heading2"]))
    story.append(Spacer(1, 12))

    designs = f"""The following figure shows in the upper panel the optimisation score
over the different generations.
Below are the expected signals of the best designs from different families,
more specific and in relation with the covariance matrix,
designs {str(population.out)[1:-1]}.
Next to each design is the covariance matrix between the regressors,
and the diagonalmatrix with the eigenvalues of the design matrix."""
    story.append(Paragraph(designs, styles["Normal"]))

    plt.figure(figsize=(12, 18))
    gs = gridspec.GridSpec(population.outdes + 4, 5)
    plt.subplot(gs[:2, :])
    plt.plot(population.optima)

    for des in range(population.outdes):
        design = population.designs[population.out[des]]
        stdes = des + 2
        plt.subplot(gs[stdes, :3])
        plt.plot(design.Xconv, lw=2)
        plt.axis("off")
        plt.subplot(gs[stdes, 3])
        varcov = np.corrcoef(design.Xconv.T)
        plt.imshow(varcov, interpolation="nearest", clim=(-1, 1), cmap="RdBu")
        plt.axis("off")
        plt.colorbar(ticks=[-1, 0, 1])
        plt.subplot(gs[stdes, 4])
        eigenv = np.diag(np.linalg.svd(design.Xconv.T)[1])
        plt.imshow(eigenv, interpolation="nearest", clim=(0, 1))
        plt.axis("off")
        plt.colorbar(ticks=[0, 1])

    story.append(PageBreak())

    intro = "Experimental settings"
    story.append(Paragraph(intro, styles["Heading2"]))
    story.append(Spacer(1, 12))

    exp = [
        ["Repetition time (TR):", population.exp.TR],
        ["Number of trials:", population.exp.n_trials],
        ["Number of scans:", population.exp.n_scans],
        ["Number of different stimuli:", population.exp.n_stimuli],
        [],
        ["Stimulus probabilities:", Table([population.exp.P], rowHeights=13)],
        [],
        ["Duration of stimulus (s)", population.exp.stim_duration],
        ["Seconds before stimulus (in trial):", population.exp.t_pre],
        ["Seconds after stimulus (in trial)", population.exp.t_post],
        ["Duration of trial (s):", population.exp.trial_duration],
        ["Total experiment duration(s):", population.exp.duration],
        [],
        ["Number of stimuli between rest blocks", population.exp.restnum],
        ["Duration of rest blocks (s):", population.exp.restdur],
        [],
        [],
        [],
        [],
        [],
        ["Contrasts:", Table(list(population.exp.C), rowHeights=13)],
        [],
        ["ITI model:", population.exp.ITImodel],
        ["minimum ITI:", population.exp.ITImin],
        ["mean ITI:", population.exp.ITImean],
        ["maximum ITI:", population.exp.ITImax],
        [],
        ["Hard probabilities: ", population.exp.hardprob],
        ["Maximum number of repeated stimuli:", population.exp.maxrep],
        ["Resolution of design:", population.exp.resolution],
        [],
        ["Assumed autocorrelation:", population.exp.rho],
    ]

    story.append(Table(exp, rowHeights=13))

    optset = "Optimalisation settings"
    story.append(Paragraph(optset, styles["Heading2"]))
    story.append(Spacer(1, 12))

    opt = [
        [
            "Optimalisation weights (Fe,Fd,Fc,Ff):",
            Table([population.weights], rowHeights=13),
        ],
        [],
        ["Aoptimality?", population.Aoptimality],
        ["Number of designs in each generation:", population.G],
        ["Number of immigrants in each generation:", population.I],
        ["Confounding order:", population.exp.confoundorder],
        ["Convergence criterion:", population.convergence],
        ["Number of precycles:", population.preruncycles],
        ["Number of cycles:", population.cycles],
        ["Percentage of mutations:", population.q],
        ["Seed:", population.seed],
    ]

    story.append(Table(opt, rowHeights=13))

    doc.build(story)


def _form_xo_reader(imgdata):
    (page,) = PdfReader(imgdata).pages
    return pagexobj(page)


class PdfImage(Flowable):
    def __init__(self, img_data, width: int = 200, height: int = 200):
        self.img_width = width
        self.img_height = height
        self.img_data = img_data

    def wrap(self):
        return self.img_width, self.img_height

    def drawOn(self, canv, x, y, _sW=0):
        # TODO TA_CENTER TA_RIGHT TA_LEFT are undefined
        if _sW > 0 and hasattr(self, "hAlign"):
            a = self.hAlign
            if a in ("CENTER", "CENTRE", TA_CENTER):
                x += 0.5 * _sW
            elif a in ("RIGHT", TA_RIGHT):
                x += _sW
            elif a not in ("LEFT", TA_LEFT):
                raise ValueError(f"Bad hAlign value {str(a)}")
        canv.saveState()
        img = self.img_data
        if isinstance(img, PdfDict):
            xscale = self.img_width / img.BBox[2]
            yscale = self.img_height / img.BBox[3]
            canv.translate(x, y)
            canv.scale(xscale, yscale)
            canv.doForm(makerl(canv, img))
        else:
            canv.drawImage(img, x, y, self.img_width, self.img_height)
        canv.restoreState()
