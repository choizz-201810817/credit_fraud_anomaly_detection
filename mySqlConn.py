#%%
import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine

# DataFrame을 mysql의 electricitydb에 table로 넣기
def df2table(df, userId='', userPw='', dbName='', tableName=''):
    db_connection_str = f'mysql+pymysql://{userId}:{userPw}@localhost/{dbName}'
    db_connection = create_engine(db_connection_str)
    df.to_sql(name=tableName, con=db_connection, if_exists='fail', index=False)

# mySql 연동하여 table을 dataframe으로 가져오기
def table2df(userId='', userPw='', dbName='', tableName=''):
    conn = pymysql.connect(host='localhost', user=userId, passwd=userPw, db=dbName, charset='utf8')
    df = pd.read_sql(f'SELECT * FROM {tableName}', con=conn)
    return df

# %%
