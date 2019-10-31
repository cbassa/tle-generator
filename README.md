# tle-generator

This is a set of *python* programs to generate two-line elements from (visual) observations.

## Background

This software is supposed to ingest visual observations of satellites via common [exchange formats](http://www.satobs.org/position/posn_formats.html)
(currently only the IOD format is supported), store them in an aggregated manner together with the required metadata (e.g. the observer location)
and create new TLEs or refine already existing TLEs using this data.

## Installation

stvid handles dependencies using pip. You can install requirements by running:
```
pip install -r requirements.txt
```

Consider using a VirtualEnv to run stvid on a separate python virtual environment.

# Configuration

- Copy the `configuration.ini-dist` file to `configuration.ini`
- Edit `configuration.ini` with your preferred settings

## File structure

- Generate the expected file structure with `init_paths.py`


## Example usage

- Read an observation file and plot the brightness vs time:
  ```
  ./read_iod.py examples/obs_20190507.dat
  ```

- Import new observations (IOD format) into storage:
  ```
  ./ingest.py examples/obs_20190507.dat
  ```

## License
&copy; 2019 Cees Bassa

Licensed under the [GPLv3 or later](LICENSE).
