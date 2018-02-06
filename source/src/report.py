from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image,Table,PageBreak, Flowable
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import letter
import matplotlib.gridspec as gridspec
from pdfrw import PdfReader, PdfDict
from pdfrw.buildxobj import pagexobj
from pdfrw.toreportlab import makerl
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
from pdfrw import PdfReader
import sklearn.cluster
import neurodesign
import numpy as np
import matplotlib
import cStringIO
import seaborn
import time
import os

plt.switch_backend('agg')

def make_report(POP,outfile="NeuroDesign.pdf"):
    '''
    Function to create a report of a finished design optimisation.
    '''
    if not type(POP.cov)==np.ndarray:
        POP.evaluate()

    styles=getSampleStyleSheet()

    doc = SimpleDocTemplate(outfile,pagesize=letter,
                            rightMargin=40,leftMargin=40,
                            topMargin=40,bottomMargin=18)

    Story=[]
    logofile = os.path.join(neurodesign.__path__[0],"media/NeuroDes.png")
    im = Image(logofile, 1*inch, 1.25*inch)
    Story.append(im)
    Story.append(Spacer(1, 12))

    title='NeuroDesign: optimalisation report'
    Story.append(Paragraph(title,styles['title']))
    Story.append(Spacer(1, 12))

    formatted_time = time.ctime()
    ptext = 'Document created: %s' % formatted_time
    Story.append(Paragraph(ptext, styles["Normal"]))
    Story.append(Spacer(1, 12))

    Story.append(Paragraph("Correlation between designs", styles["Heading2"]))
    Story.append(Spacer(1, 12))

    corr='During the optimisation, the designs are mixed with each other to find better combinations.  As such, the designs can look very similar. Actually, the genetic algorithm uses natural selection as a basis, and as such, the designs can be clustered in families.  This is the covariance matrix between the final {0} designs'.format(POP.G)
    Story.append(Paragraph(corr, styles["Normal"]))

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(POP.cov,interpolation="nearest")
    plt.colorbar()
    imgdata = cStringIO.StringIO()
    fig.savefig(imgdata, format='pdf')
    imgdata.seek(0)  # rewind the data

    reader = form_xo_reader
    image = reader(imgdata)
    img = PdfImage(image,width=300,height=250)
    Story.append(img)

    Story.append(PageBreak())

    Story.append(Paragraph("Selected designs", styles["Heading2"]))
    Story.append(Spacer(1, 12))

    designs='The following figure shows in the upper panel the optimisation score over the different generations.  Below are the expected signals of the best designs from different families, more specific and in relation with the covariance matrix, designs {0}.  Next to each design is the covariance matrix between the regressors, and the diagonalmatrix with the eigenvalues of the design matrix.'.format(str(POP.out)[1:-1])
    Story.append(Paragraph(designs, styles["Normal"]))

    fig = plt.figure(figsize=(12, 18))
    gs = gridspec.GridSpec(POP.outdes+4,5)
    plt.subplot(gs[:2,:])
    plt.plot(POP.optima)

    for des in range(POP.outdes):
        design = POP.designs[POP.out[des]]
        stdes = des+2
        plt.subplot(gs[stdes,:3])
        plt.plot(design.Xconv,lw=2)
        plt.axis('off')
        plt.subplot(gs[stdes,3])
        varcov = np.corrcoef(design.Xconv.T)
        plt.imshow(varcov,interpolation='nearest',clim=(-1,1),cmap="RdBu")
        plt.axis('off')
        plt.colorbar(ticks=[-1,0,1])
        plt.subplot(gs[stdes,4])
        eigenv = np.diag(np.linalg.svd(design.Xconv.T)[1])
        plt.imshow(eigenv,interpolation='nearest',clim=(0,1))
        plt.axis('off')
        plt.colorbar(ticks=[0,1])

    imgdata = cStringIO.StringIO()
    fig.savefig(imgdata, format='pdf')
    imgdata.seek(0)  # rewind the data

    reader = form_xo_reader
    image = reader(imgdata)
    img = PdfImage(image,width=500,height=600)
    Story.append(img)

    Story.append(PageBreak())

    intro='Experimental settings'
    Story.append(Paragraph(intro, styles["Heading2"]))
    Story.append(Spacer(1, 12))

    exp = [
        ["Repetition time (TR):",POP.exp.TR],
        ["Number of trials:",POP.exp.n_trials],
        ["Number of scans:",POP.exp.n_scans],
        ["Number of different stimuli:",POP.exp.n_stimuli],
        [],
        ["Stimulus probabilities:",Table([POP.exp.P],rowHeights=13)],
        [],
        ["Duration of stimulus (s)",POP.exp.stim_duration],
        ["Seconds before stimulus (in trial):",POP.exp.t_pre],
        ["Seconds after stimulus (in trial)",POP.exp.t_post],
        ["Duration of trial (s):",POP.exp.trial_duration],
        ["Total experiment duration(s):",POP.exp.duration],
        [],
        ["Number of stimuli between rest blocks",POP.exp.restnum],
        ["Duration of rest blocks (s):",POP.exp.restdur],
        [],
        [],
        [],
        [],
        [],
        ["Contrasts:",Table(list(POP.exp.C),rowHeights=13)],
        [],
        ["ITI model:",POP.exp.ITImodel],
        ["minimum ITI:",POP.exp.ITImin],
        ["mean ITI:",POP.exp.ITImean],
        ["maximum ITI:",POP.exp.ITImax],
        [],
        ["Hard probabilities: ",POP.exp.hardprob],
        ["Maximum number of repeated stimuli:",POP.exp.maxrep],
        ["Resolution of design:",POP.exp.resolution],
        [],
        ["Assumed autocorrelation:",POP.exp.rho]
    ]

    Story.append(Table(exp,rowHeights=13))

    optset='Optimalisation settings'
    Story.append(Paragraph(optset, styles["Heading2"]))
    Story.append(Spacer(1, 12))

    opt = [
        ["Optimalisation weights (Fe,Fd,Fc,Ff):",Table([POP.weights],rowHeights=13)],
        [],
        ["Aoptimality?",POP.Aoptimality],
        ["Number of designs in each generation:",POP.G],
        ["Number of immigrants in each generation:",POP.I],
        ["Confounding order:",POP.exp.confoundorder],
        ["Convergence criterion:",POP.convergence],
        ["Number of precycles:",POP.preruncycles],
        ["Number of cycles:",POP.cycles],
        ["Percentage of mutations:",POP.q],
        ["Seed:",POP.seed]
    ]

    Story.append(Table(opt,rowHeights=13))

    doc.build(Story)

def form_xo_reader(imgdata):
    page, = PdfReader(imgdata).pages
    return pagexobj(page)

class PdfImage(Flowable):
    def __init__(self, img_data, width=200, height=200):
        self.img_width = width
        self.img_height = height
        self.img_data = img_data

    def wrap(self, width, height):
        return self.img_width, self.img_height

    def drawOn(self, canv, x, y, _sW=0):
        if _sW > 0 and hasattr(self, 'hAlign'):
            a = self.hAlign
            if a in ('CENTER', 'CENTRE', TA_CENTER):
                x += 0.5*_sW
            elif a in ('RIGHT', TA_RIGHT):
                x += _sW
            elif a not in ('LEFT', TA_LEFT):
                raise ValueError("Bad hAlign value " + str(a))
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
