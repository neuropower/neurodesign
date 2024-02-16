.. _getting_started:

***************
Getting started
***************

.. _installing-neurodesign:

Installing NeuroDesign
======================

neurodesign is available on pypi.  To install, run::

  pip install neurodesign

.. _about-designopt:

About Design Optimisation Using the Genetic Algorithm
=====================================================

This toolbox is for the optimization of experimental designs for fMRI.
Minimizing the variance of the design matrix will help detect or estimate
(depending on the outcome of interest) the effect researchers are looking for.
The genetic algorithm for experimental designs was introduced by Wager and Nichols (2002)
and further improved by Kao, Mandal, Lazar and Stufken (2009).
We implemented these methods in a python package and a userfriendly web-application
and introduced some improvements and allows more flexibility for the experimental setup.

.. _design-efficiency:

Design efficiency
=================

The core idea of this package is to run an optimization algorithm that (among others)
optimizes the design efficiency of an fMRI design using A-optimality, with this formula:

.. math::
   Eff(C\beta) = \frac{n}{Trace(CX'X^{-1}C')}
