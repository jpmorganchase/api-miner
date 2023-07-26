import json
from pathlib import Path
import sqlite3
from api_miner.configs.db_configs import DB_PATH
con = sqlite3.connect(DB_PATH, check_same_thread=False)

def setup_db():
    cur = con.cursor()
    cur.execute("create table if not exists api_table (endpoint blob , api_content blob)")
    con.commit()
    cur.close()

def close_db():
    if con:
        con.close()

def append_to_disk(endpoint_name: str, endpoint_obj):
    """
    Append endpoint object to database 
    """
    try:
        cur = con.cursor()
        byte_key = endpoint_name.encode('utf8')

        # Get existing api_content for byte_key
        cur.execute("select api_content from api_table where endpoint=:e", {"e": byte_key})
        byte_content = cur.fetchall()[0][0]
        content = json.loads(byte_content.decode("utf8"))
        
        # Append new api_content to exsting api_content (as array)
        content.append(endpoint_obj._asdict())
        new_byte_content = json.dumps(content, default = set_to_list).encode("utf8")
        cur.execute("update api_table set api_content = ? where endpoint = ?", (new_byte_content, byte_key))

        con.commit()
        cur.close()

    except Exception as e:
        print(byte_key)
        raise Exception("Appending to disk failed for api endpoint: ", endpoint_name)

def write_to_disk(endpoint_name: str, endpoint_obj):
    """
    Write endpoint object to disk
    """
    try:
        cur = con.cursor()
        byte_key = endpoint_name.encode('utf8')
        byte_content = json.dumps([endpoint_obj._asdict()], default = set_to_list).encode("utf8")
        cur.execute("insert into api_table values (?, ?)", (byte_key, byte_content))
        con.commit()
        cur.close()

    except Exception as e:
        raise Exception("Writing to disk failed for api endpoint: ", endpoint_name)

def set_to_list(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def check_db_count():
    '''
    Check estimate of number of items in db
    For debugging / testing
    '''
    cur = con.cursor()
    cur.execute("select count(*) from api_table")
    num_est = cur.fetchone()[0]
    print('Writing to db count est: ', num_est)
    cur.close()

def load_from_disk(endpoints, as_str=True):
    """
    Return all API specs that contain at least one of the endpoints in endpoints
    """
    specs = []
    for endpoint in endpoints:
        cur = con.cursor()
        
        byte_key = endpoint.encode('utf8')
        cur.execute("select api_content from api_table where endpoint=:e", {"e": byte_key})
        byte_content = cur.fetchall()[0][0]

        if not byte_content:
            continue
        
        content = byte_content.decode('utf8')
        if not as_str:
            content = json.loads(content)

        specs.append(content)
    return specs