import pymysql
from pymysql.cursors import DictCursor
from pymysql.connections import Connection

import os
import logging

logger = logging.getLogger(__name__)

# Database configuration (use environment variables; defaults target XAMPP local setup)
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', '127.0.0.1'),
    'user': os.environ.get('DB_USER', 'root'),
    # XAMPP default: root user with empty password
    'password': os.environ.get('DB_PASSWORD', ''),
    'db': os.environ.get('DB_NAME', 'soccer_analytics'),
    'charset': 'utf8mb4',
    'cursorclass': DictCursor
}

def get_db():
    """Dependency function to get a database connection and ensure it's closed."""
    conn = None
    try:
        conn = pymysql.connect(**DB_CONFIG)
        yield conn
    except pymysql.Error as e:
        logger.exception("Error connecting to database")
        raise
    finally:
        if conn:
            conn.close()
