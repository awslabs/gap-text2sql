import os
import string
import sqlite3
import nltk.corpus
import re
from seq2struct.resources import corenlp

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
PUNKS = set(a for a in string.punctuation)


# schema linking, similar to IRNet
def compute_schema_linking(question, column, table):

    def partial_match(x_list, y_list):
        x_str = " ".join(x_list)
        y_str = " ".join(y_list)
        if x_str in STOPWORDS or x_str in PUNKS:
            return False
        if re.match(rf"\b{re.escape(x_str)}\b", y_str):
            assert x_str in y_str
            return True
        else:
            return False

    def exact_match(x_list, y_list):
        x_str = " ".join(x_list)
        y_str = " ".join(y_list)
        if x_str == y_str:
            return True
        else:
            return False

    q_col_match = dict()
    q_tab_match = dict()

    col_id2list = dict()
    for col_id, col_item in enumerate(column):
        if col_id == 0:
            continue
        col_id2list[col_id] = col_item
        
    tab_id2list = dict()
    for tab_id, tab_item in enumerate(table):
        tab_id2list[tab_id] = tab_item

    # 5-gram
    n = 5
    while n > 0:
        for i in range(len(question) - n + 1):
            n_gram_list = question[i:i + n]
            n_gram = " ".join(n_gram_list)
            if len(n_gram.strip()) == 0:
                continue
            # exact match case
            for col_id in col_id2list:
                if exact_match(n_gram_list, col_id2list[col_id]):
                    for q_id in range(i, i+n):
                        q_col_match[f"{q_id},{col_id}"] = "CEM"
            for tab_id in tab_id2list:
                if exact_match(n_gram_list, tab_id2list[tab_id]):
                    for q_id in range(i, i+n):
                        q_tab_match[f"{q_id},{tab_id}"] = "TEM"

            # partial match case
            for col_id in col_id2list:
                if partial_match(n_gram_list, col_id2list[col_id]):
                    for q_id in range(i, i+n):
                        if f"{q_id},{col_id}" not in q_col_match:
                            q_col_match[f"{q_id},{col_id}"] = "CPM"
            for tab_id in tab_id2list:
                if partial_match(n_gram_list, tab_id2list[tab_id]):
                    for q_id in range(i, i+n):
                        if f"{q_id},{tab_id}" not in q_tab_match:
                            q_tab_match[f"{q_id},{tab_id}"] = "TPM"
        n -= 1
    return {"q_col_match": q_col_match, "q_tab_match": q_tab_match }


def compute_cell_value_linking(tokens, schema, db_dir):

    def isnumber(word):
        try:
            float(word)
            return True
        except:
            return False

    def db_word_match(word, column, table, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        p_str = f"select {column} from {table} where {column} like '{word} %' or {column} like '% {word}' or {column} like '% {word} %'  or {column} like '{word}'" 
        try:
            cursor.execute(p_str)
            p_res = cursor.fetchall()
            if len(p_res) == 0:
                return False
            else:
                return p_res
        except:
            return False

    db_name = schema.db_id
    db_path = os.path.join(db_dir, db_name, db_name + '.sqlite')

    num_date_match = {}
    cell_match = {}

    for q_id, word in enumerate(tokens):
        if len(word.strip()) == 0:
            continue
        if word in STOPWORDS or word in PUNKS:
            continue 

        num_flag = isnumber(word) 

        for col_id, column in enumerate(schema.columns):
            if col_id == 0: 
                assert column.orig_name == "*"
                continue
            
            # word is number 
            if num_flag:
                if column.type in ["number", "time"]: # TODO fine-grained date
                    num_date_match[f"{q_id},{col_id}"] = column.type.upper()
            else: 
                ret = db_word_match(word, column.orig_name, column.table.orig_name, db_path)
                if ret: 
                    # print(word, ret)
                    cell_match[f"{q_id},{col_id}"] = "CELLMATCH" 

    cv_link = {"num_date_match": num_date_match, "cell_match" : cell_match, "normalized_token": tokens}
    return cv_link