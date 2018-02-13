0.2.01 - 02/09/2018
* bug in report.py

0.2.00 - 02/09/2018
* change names to reflect new options
* update examples
* reflects JSS submission

0.1.13 - 02/06/2018
* fix on not doing FdMax and FeMax if not necessary

0.1.12 - 02/06/2018
* fix on matplotlib bug

0.1.08-11 - 02/06/2018
* fix on duration bug

0.1.07 - 02/05/2018
* added option for random design generation

0.1.06 - 01/22/2018
* debug conda matplotlib
* remove shift in onsets
* ensure ITI is nominal value
* improve handling of resolution
* improve handling of hrf (resolution)

0.1.05 - 11/22/2017
* report: remove bug ndarray check in report

0.1.04 - 08/30/2017
* add hrf_precision as parameter: not same as resolution

0.1.03 - 08/09/2017
* remove indentation bugs
* replace often reoccuring warnings with internal warning

0.1.02 - 07/31/2017
* debug first ITI (should be 0)
* debug Hardprob
* debug maximum ITI with resolution
* round ITI to resolution before computing onsets
* added exceptions for longer designs

0.1.01 - 07/03/2017
* debug numpy exception

0.1.00 - 04/26/2017
* package underwent extensive testing through the online application
* examples are added complementary to the manuscript under revision

0.0.28 - 02/20/2017
* debug correlation check

0.0.27 - 02/20/2017
* debug rest durations
* debug repeated msequences

0.0.26 - 02/14/2017
* patch reoccurring (but rare) problem with too long a design

0.0.25 - 02/06/2017
* avoid getting stuck while trying to find blocked designs

0.0.24 - 02/02/2017
* add some extra space for max duration (0.5s) to avoid trials exceeding the experiment duration

0.0.23 - 02/01/2017
* remove requirements for now

0.0.22 - 02/01/2017
* update requirements

0.0.21 - 02/01/2017
* remove itertools from requirements

0.0.20 - 02/01/2017
* remove collections from requirements

0.0.19 - 02/01/2017
* pypi problems

0.0.18 - 02/01/2017
* remove typo in install_requires

0.0.17 - 02/01/2017
* remove requirements.txt

0.0.16 - 02/01/2017
* debug memory problem induced in 0.0.15

0.0.15 - 01/31/2017
* debug LinAlgError
* debug necessity for calculating both Fe and Fd --> speed up
* debug initial population weights

0.0.14 - 01/19/2017
* improve which designs are outputted
* improve report

0.0.13 - 01/18/2017
* debug max number of repeats

0.0.12 - 01/18/2017
* debug generation of ITI's from truncated distribution

0.0.11 - 01/17/2017
* debug generation of ITI's from truncated distribution

0.0.10 - 01/10/2017
* remove direct support for GUI

0.0.9 - 12/19/2016
* change matplotlib backend for report

0.0.8 - 12/19/2016
* debug script

0.0.7 - 12/09/2016
* debug seeds
* restructure a few parameters in other classes

0.0.6 - 11/18/2016
* add report to download

0.0.2 - 11/18/2016
* remove bug in report when numpy array is given.
