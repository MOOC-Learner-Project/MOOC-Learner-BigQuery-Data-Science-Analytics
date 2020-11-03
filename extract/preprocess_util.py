import hashlib
import os
from collections import namedtuple
from functools import lru_cache


def in_ipython():
    """Boolean function check for ipython."""
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


@lru_cache(maxsize=None)
def hash_username(n):
    """Hashing function for usernames."""
    return hashlib.md5(n.encode('utf-8')).hexdigest()


def get_datetime_parser():
    """Makes a parser for datetime strings."""
    try:
        import ciso8601

        def parser(time_string):
            if time_string is None:
                return None
            return ciso8601.parse_datetime(time_string[:-4])
        return parser
    except ImportError:
        import datetime

        def parser(time_string):
            if time_string is None:
                return None
            try:
                return datetime.datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S.%f UTC")
            except ValueError:
                return datetime.datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S UTC")
        return parser