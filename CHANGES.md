# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security
-->

## [Unreleased]

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.2.02] - 2018-08-24

* bug with timepoints fix

## [0.2.01] - 2018-02-09

* bug in report.py

## [0.2.00] - 2018-02-09

* change names to reflect new options
* update examples
* reflects JSS submission

## [0.1.13] - 2018-02-06

* fix on not doing FdMax and FeMax if not necessary

## [0.1.12] - 2018-02-06

* fix on matplotlib bug

## [0.1.08]-11 - 2018-02-06

* fix on duration bug

## [0.1.07] - 2018-02-05

* added option for random design generation

## [0.1.06] - 2018-11-22

* debug conda matplotlib
* remove shift in onsets
* ensure ITI is nominal value
* improve handling of resolution
* improve handling of hrf (resolution)

## [0.1.05] - 2017-11-22

* report: remove bug ndarray check in report

## [0.1.04] - 2017-08-30

* add hrf_precision as parameter: not same as resolution

## [0.1.03] - 2017-08-09

* remove indentation bugs
* replace often reoccuring warnings with internal warning

## [0.1.02] - 2017-07-31

* debug first ITI (should be 0)
* debug Hardprob
* debug maximum ITI with resolution
* round ITI to resolution before computing onsets
* added exceptions for longer designs

## [0.1.01] - 2017-07-03

* debug numpy exception

## [0.1.00] - 2017-04-26

* package underwent extensive testing through the online application
* examples are added complementary to the manuscript under revision

## [0.0.28] - 2017-02-20

* debug correlation check

## [0.0.27] - 2017-02-20

* debug rest durations
* debug repeated msequences

## [0.0.26] - 2017-02-14

* patch reoccurring (but rare) problem with too long a design

## [0.0.25] - 2017-02-06

* avoid getting stuck while trying to find blocked designs

## [0.0.24] - 2017-02-02

* add some extra space for max duration (0.5s) to avoid trials exceeding the experiment duration

## [0.0.23] - 2017-02-01

* remove requirements for now

## [0.0.22] - 2017-02-01

* update requirements

## [0.0.21] - 2017-02-01

* remove itertools from requirements

## [0.0.20] - 2017-02-01

* remove collections from requirements

## [0.0.19] - 2017-02-01

* pypi problems

## [0.0.18] - 2017-02-01

* remove typo in install_requires

## [0.0.17] - 2017-02-01

* remove requirements.txt

## [0.0.16] - 2017-02-01

* debug memory problem induced in 0.0.15

## [0.0.15] - 2017-01-31

* debug LinAlgError
* debug necessity for calculating both Fe and Fd --> speed up
* debug initial population weights

## [0.0.14] - 2017-01-19

* improve which designs are outputted
* improve report

## [0.0.13] - 2017-01-18

* debug max number of repeats

## [0.0.12] - 2017-01-18

* debug generation of ITI's from truncated distribution

## [0.0.11] - 2017-01-17

* debug generation of ITI's from truncated distribution

## [0.0.10] - 2017-01-10

* remove direct support for GUI

## [0.0.9] - 2016-12-19

* change matplotlib backend for report

## [0.0.8] - 2016-12-19

* debug script

## [0.0.7] - 2016-12-09

* debug seeds
* restructure a few parameters in other classes

## [0.0.6] - 2016-11-18

* add report to download

## [0.0.2] - 2016-11-18

* remove bug in report when numpy array is given.
