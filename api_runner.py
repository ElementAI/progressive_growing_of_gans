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
    s3_bucket_name = os.environ.get('S3_BUCKET_NAME', '')
    if s3_bucket_name:
        Config.set('s3_bucket_name', s3_bucket_name)
    s3_directory = os.environ.get('S3_DIRECTORY', '')
    if s3_bucket_name:
        Config.set('s3_directory', s3_directory)
    # twitter credentials
    consumer_key = os.environ.get('CONSUMER_KEY', '')
    if consumer_key:
        Config.set('consumer_key', consumer_key)
    consumer_secret = os.environ.get('CONSUMER_SECRET', '')
    if consumer_secret:
        Config.set('consumer_secret', consumer_secret)
    access_token = os.environ.get('ACCESS_TOKEN', '')
    if access_token:
        Config.set('access_token', access_token)
    access_token_secret = os.environ.get('ACCESS_TOKEN_SECRET', '')
    if access_token_secret:
        Config.set('access_token_secret', access_token_secret)
    bitly_access_token = os.environ.get('BITLY_ACCESS_TOKEN', '')
    if bitly_access_token:
        Config.set('bitly_access_token', bitly_access_token)
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
    s3_bucket_name = os.environ.get('S3_BUCKET_NAME', '')
    if s3_bucket_name:
        Config.set('s3_bucket_name', s3_bucket_name)
    s3_directory = os.environ.get('S3_DIRECTORY', '')
    if s3_directory:
        Config.set('s3_directory', s3_directory)
    # twitter credentials
    consumer_key = os.environ.get('CONSUMER_KEY', '')
    if consumer_key:
        Config.set('consumer_key', consumer_key)
    consumer_secret = os.environ.get('CONSUMER_SECRET', '')
    if consumer_secret:
        Config.set('consumer_secret', consumer_secret)
    access_token = os.environ.get('ACCESS_TOKEN', '')
    if access_token:
        Config.set('access_token', access_token)
    access_token_secret = os.environ.get('ACCESS_TOKEN_SECRET', '')
    if access_token_secret:
        Config.set('access_token_secret', access_token_secret)
    bitly_access_token = os.environ.get('BITLY_ACCESS_TOKEN', '')
    if bitly_access_token:
        Config.set('bitly_access_token', bitly_access_token)
    api.init()
    api.app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()
