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

- Import new observations (IOD format) into storage:
  ```
  ./ingest.py examples/example.dat
  ```
- Update TLE for object 37386 with:
  ```
  ./preprocess.py -c elements/37386.txt -d observations/37386.dat -i 37386
  ```

## License
&copy; 2020 Cees Bassa

Licensed under the [GPLv3 or later](LICENSE).
