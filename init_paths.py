#!/usr/bin/env python3

import os
import configparser


if __name__ == '__main__':
    cfg = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    cfg.read('configuration.ini')

    # Generate directory
    if not os.path.exists(cfg.get('Common', 'observations_path')):
        os.makedirs(cfg.get('Common', 'observations_path'))
