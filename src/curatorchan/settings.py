# Curator-chan
# Copyright (C) 2024  Oliver Harcourt

# This file is part of Curator-chan.

# Curator-chan is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Curator-chan is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with Curator-chan.  If not, see <https://www.gnu.org/licenses/>.

import logging
from logging.config import dictConfig

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "%(asctime)s [%(levelname)s] %(module)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "standard": {"format": "[%(levelname)s] - %(module)s : %(message)s"},
        "debug": {"format": "[%(levelname)s] - (%(name)s) - %(module)s : %(message)s"},
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "debug",
        },
        "console2": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
        "file": {
            "level": "INFO",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/curatorchan.log",
            "formatter": "verbose",
            "mode": "a",
            "maxBytes": 10485760,
            "backupCount": 5,
        },
    },
    "loggers": {
        "curatorchan": {
            "handlers": ["file"],
            "level": "INFO",
            "propagate": False,
        },
        "bot-dev": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}

dictConfig(LOGGING_CONFIG)
