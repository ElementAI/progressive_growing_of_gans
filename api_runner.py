#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import os
import argparse
import logging
from utils.config import Config
import api

def create_parser_args():
    parser = argparse.ArgumentParser(
            description='Progressive Growing of GANs API',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            )
    parser.add_argument('--cache', '-a', action='store_true',
                        help='Enable cache')
    parser.add_argument('--config', '-c', default='', type=str,
                        help='Config file')
    return parser

def gunicorn():
    logging.basicConfig(level=logging.DEBUG)

    config_path = os.environ.get('CONFIG', '')
    cache = os.environ.get('CACHE', '')
    if config_path:
        Config.load_file(config_path)
    Config.set('cache', cache in [True, '1', 'true', 'True'])
    api.init()

    return api.app


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = create_parser_args()
    args = parser.parse_args()
    config_path = args.config or os.environ.get('CONFIG', '')
    if config_path:
        Config.load_file(config_path)
    cache = args.cache or os.environ.get('CACHE', '') in [True, '1', 'true', 'True']
    Config.set('cache', cache)

    api.init()
    api.app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()
