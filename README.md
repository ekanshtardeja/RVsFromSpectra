# RVsFromSpectra

**Stellar Motion** is a Python-based pipeline designed to automatically calculate radial velocities of stars from their observed spectra by analysing absorption-line shifts.



## Workflow

1. Enter the name of a star.
2. Retrieve the available stellar spectra from the supported archive.
3. Specify the number of absorption lines and their laboratory rest wavelengths.
4. Plot the selected spectral regions and fit Gaussian profiles to the absorption lines.
5. Calculate the radial velocity for each selected line.
6. Calculate the average radial velocity for each spectrum.
7. Repeat the analysis for all retrieved spectra and compile the resulting radial velocity measurements.

## Purpose

The goal of this pipeline is to simplify and automate radial velocity measurements from stellar spectra, supporting the study of stellar motion, spectroscopic binary systems, and radial velocity variations across multiple observing epochs.

