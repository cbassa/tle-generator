# tle-generator

This is a set of python tools to generate two-line elements (TLEs) from (visual) observations.

## Background

This software is supposed to ingest visual observations of satellites via common [exchange formats](http://www.satobs.org/position/posn_formats.html)
(currently only the IOD format is supported), store them in an aggregated manner together with the required metadata (e.g. the observer location)
and create new TLEs or refine already existing TLEs using this data.

## Installation

`tle-generator` handles dependencies using `pip`. You can install requirements by running:
```
pip install -r requirements.txt
```

Consider using a VirtualEnv to run stvid on a separate python virtual environment.

# Configuration

- Copy the `configuration.ini-dist` file to `configuration.ini`
- Edit `configuration.ini` with your preferred settings

## Example usage

- Ingest new observations and/or orbital elements into the database:
  ```
  ./ingest.py -c elements/37386.txt -d observations/37386.dat
  ```
- Select observations from a satellite over a time range (30 day period upto 2019, May 16):
  ```
  ./preprocess.py -i 37386 -t 2019-05-16T00:00:00 -l 30
  ```
  The selected observations and nearest orbital elements are stored in a `yaml` file of the form:
  ```
  observations:
  - 37386 11 014A   4171 G 20190507205224671 17 25 1656431+025146 37 S
  - 37386 11 014A   4171 G 20190507205229692 17 25 1656664+022450 37 S
  ...
  - 37386 11 014A   4171 G 20190513215410498 17 25 1313244-122778 37 S
  - 37386 11 014A   4171 G 20190513215415511 17 25 1315396-125773 37 S
  tle:
    line0: NOSS 3-5 (A)
    line1: 1 37386U 11014A   19116.95390559 0.00000000  00000-0  00000-0 0    00
    line2: 2 37386  63.4392  89.1087 0131442   0.1540 359.8459 13.40775636    09
  ```
  This `yaml` file can be adjusted manually if needed.

- Compute and plot residuals:
  ```
  ./residuals.py -y 37386.yaml
  ```

- Improve the fit. This overwrites the TLE in the `yaml` file. Rerun the residual computation for an updated plot:
  ```
  ./satfit.py -y 37386.yaml
  ./residuals.py -y 37386.yaml
  ```
  
- As an option, an `mcmc` analysis can be ran to further improve the fit and investigate the covariances of the parameters:
  ```
  ./mcmc.py -y 37386.yaml
  ```

- The updated TLEs can be ingested through: 
  ```
  ./ingest.py -c 37386.txt
  ```
## License
&copy; 2019-2021 Cees Bassa

Licensed under the [GPLv3 or later](LICENSE).
