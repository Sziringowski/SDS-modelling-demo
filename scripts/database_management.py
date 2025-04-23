import psycopg2
from typing import List
import documentation as doc

int_and_text = ["id", "protection", "size", "nodes", "op_type", "hw_chassis", "ssd", "path"]
floats = ["sum_mbps", "sum_opps"]

class database:
    def __init__(self, dbname, user, password, host="localhost", port="5432"):  # коннект к бд
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port

        #
        self._create_sizes()
        self._create_synth()


    def _create_sizes(self):
        with psycopg2.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        ) as connection:
            with connection.cursor() as cur:
                cur.execute("""
                CREATE TABLE IF NOT EXISTS sizes (
                    id TEXT PRIMARY KEY,
                    protection TEXT,
                    size INT,
                    nodes INT,
                    op_type TEXT,
                    hw_chassis TEXT,
                    ssd INT,
                    mbps REAL[],
                    avg REAL[],
                    min REAL[],
                    med REAL[],
                    max REAL[],
                    p90 REAL[],
                    p95 REAL[],
                    opps REAL[],
                    opps_loss REAL[],
                    sum_mbps REAL,
                    sum_opps REAL,
                    path TEXT
                )
                """)
                connection.commit()

    def _create_synth(self):
        with psycopg2.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        ) as connection:
            with connection.cursor() as cur:
                cur.execute("""
                CREATE TABLE IF NOT EXISTS sizes_synthetic (
                    id TEXT PRIMARY KEY,
                    protection TEXT,
                    size INT,
                    nodes INT,
                    op_type TEXT,
                    hw_chassis TEXT,
                    ssd INT,
                    sum_opps REAL,
                    r2 REAL,
                    mse REAL,
                    mae REAL   
                )
                """)
                connection.commit()

    def PUT(self, values: dict, table_name: str, validation=True) -> str:
        #, id: str, protection: str, size: int, nodes: int, op_type: str, hw_chassis: str, ssd: int, mbps: List[float], avg: List[float], min: List[float], med: List[float], max: List[float], p90: List[float], p95: List[float], opps: List[float], opps_loss: List[float], sum_mbps: float, sum_opps: float, path: str
        features_to_check = {
            "protection": values["protection"],
            "size": values["size"], 
            "nodes": values["nodes"], 
            "op_type": values["op_type"], 
            "hw_chassis": values["hw_chassis"], 
            "ssd": values["ssd"], 
            "sum_opps": values["sum_opps"]
        }
        
        if (not validation) or (len(self.GET(targets=["id"], features=features_to_check, table_name=table_name)) == 0):
            try:
                with psycopg2.connect(
                    dbname=self.dbname,
                    user=self.user,
                    password=self.password,
                    host=self.host,
                    port=self.port
                ) as connection:
                    with connection.cursor() as cur:
                        query = f'''
                        INSERT INTO \"{table_name}\"
                        ({', '.join(values.keys())})
                        VALUES ({', '.join(['%s'] * len(values))})
                        '''
                        cur.execute(query, tuple(values.values()))
                        connection.commit()

            except Exception as e:
                print(f"Error occurred: {e}")
        else: pass
    

    def GET(self, targets: List[str], features: dict, table_name: str, distinct=False):
        '''
        `targets` списком загоняем желаемые атрибуты вывода
        `features` - то, по чему будем сортировать в SQL запросе, формат JSON
        `table_name`
        `distinct`
        '''

        '''
        if any([key not in doc.features + ["id"] for key in features.keys()]):
            return "wrong attribute"
        '''
        if False: return 0
        else: 
            targets_to_sql = ', '.join(targets) if len(targets)!=0 else '*'
            with psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            ) as connection:
                with connection.cursor() as cur:
                    if len(features)!=0:
                        conditions_1 = [f"{key} = %s" for key in features.keys() if key in int_and_text] # для целых и строковых
                        conditions_2 = [f"ABS({key} - %s) <= 0.001" for key in features.keys() if key in floats] # для флоатов
                        conditions = ' AND '.join(conditions_1+conditions_2)
                        is_distinct = 'DISTINCT' if distinct else ''
                        query = f"SELECT {is_distinct} {targets_to_sql} FROM \"{table_name}\" WHERE {conditions}"
                        cur.execute(query, tuple(features.values()))
                        output = cur.fetchall()
                        return output
                    else:
                        is_distinct = 'DISTINCT' if distinct else ''
                        query = f"SELECT {is_distinct} {targets_to_sql} FROM \"{table_name}\""
                        cur.execute(query)
                        output = cur.fetchall()
                        return output
                    
                    

    def DELETE(self, id: str, table_name: str):
        with psycopg2.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        ) as connection:
            with connection.cursor() as cur:
                query = f'''
                DELETE FROM \"{table_name}\"
                WHERE id = %s
                '''
                cur.execute(query, (id, ))
                connection.commit()
