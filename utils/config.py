# -*- coding: utf-8 -*-
#

import re
import json
import yaml
from datetime import datetime
import parsedatetime


def convert_duration(value):
    """ Convert an input duration in seconds
    """
    if type(value) is int:
        return value

    cal = parsedatetime.Calendar()
    date_now = datetime.now().replace(microsecond=0)
    date, _ = cal.parseDT(value, sourceTime=date_now)
    return (date - date_now).total_seconds()


configs = {
    'cache': {
        'description': 'Cache',
        'default': False,
        'type': bool
    },
    'cache_dir': {
        'description': 'Cache directory',
        'default': '/tmp/cache',
        'type': str
    },
    's3_bucket_name': {
        'description': 'S3 bucket name',
        'default': '',
        'type': str
    },
    's3_directory': {
        'description': 'S3 directory',
        'default': '',
        'type': str
    },
}


class Config:

    configs_cached = {}
    regex_config_replace = re.compile('%\w+%')

    @staticmethod
    def _parse_value(getter, value):
        if isinstance(value, str):
            for match in Config.regex_config_replace.findall(value):
                replace_value = getter(match[1:-1], None)
                if replace_value is not None:
                    if value == match:
                        value = replace_value
                        break
                    else:
                        value = value.replace(match, str(replace_value))
        return value

    @staticmethod
    def get_keys():
        return configs.keys()

    @staticmethod
    def get_default(property_name, default=None):
        """ Get default property value
        """
        prop = configs.get(property_name)
        if not prop:
            return default

        value = prop.get('default')
        value = Config._parse_value(Config.get_default, value)

        if not value and default:
            return default

        return value

    @staticmethod
    def load_file(filename):
        """ Load a file JSON or YAML
        """
        with open(filename, "r") as f:
            config_str = f.read()

        valid = False
        config_dict = {}
        try:
            config_dict = json.loads(config_str)
            valid = True
        except ValueError:
            valid = False

        if not valid:
            try:
                config_dict = yaml.safe_load(config_str)
                valid = True
            except yaml.scanner.ScannerError:
                valid = False

        if not valid or config_dict is not None and not isinstance(config_dict, dict):
            raise ValueError("Config file {} have to contain JSON or YAML content".format(filename))

        if config_dict:
            for (key, value) in config_dict.items():
                Config.set(key, value)

    @staticmethod
    def get(property_name, default=None):
        """ Get property value
        """
        if property_name in Config.configs_cached and Config.configs_cached[property_name]:
            return Config.configs_cached[property_name]

        prop = configs.get(property_name)
        if not prop:
            return default

        value = prop.get('value')
        if value is None:
            value = prop.get('default')
        value = Config._parse_value(Config.get, value)

        value = Config.convert(prop, value)
        Config.configs_cached[property_name] = value

        if not value and default:
            return default

        return value

    @staticmethod
    def set(property_name, value):
        """ Set new property value
        """
        if property_name not in configs:
            configs[property_name] = {}
        Config.clear_cache(property_name)
        configs[property_name]['value'] = value
        return True

    @staticmethod
    def clear_cache(property_name):
        """ Clear cache for property
        """
        if property_name in Config.configs_cached:
            del Config.configs_cached[property_name]

        prop_var = '%' + property_name + '%'
        for (key, prop) in configs.items():
            if key in Config.configs_cached:
                prop_value = prop.get('value')
                if prop_value is None:
                    prop_value = prop.get('default')
                if isinstance(prop_value, str) and prop_var in prop_value:
                    Config.clear_cache(key)

    @staticmethod
    def convert(prop, value):
        """ Convert value to property type
        """
        if value:
            if 'convert' in prop and callable(prop['convert']):
                value = prop['convert'](value)

            if 'type' in prop:
                type_ = prop['type']
                return type_(value)
        return value
