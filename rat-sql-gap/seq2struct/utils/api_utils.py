import re


def _process_name(name: str):
  # camelCase to spaces
  name = re.sub("([a-z])([A-Z])", "\g<1> \g<2>", name)
  return name.replace("-", " ").replace("_", " ").lower()

def _prompt_table(table_name, prompt_user=True):
  table_name = _process_name(table_name)
  print(f"Current table name: {table_name}")
  new_name = (
    input("Type new name (empty to keep previous name): ") if prompt_user else ""
  )
  return new_name if new_name != "" else table_name


def _prompt_column(column_name, table_name, prompt_user=True):
  column_name = _process_name(column_name)
  print(f"Table {table_name}. Current col name: {column_name}")
  new_name = (
    input("Type new name (empty to keep previous name): ") if prompt_user else ""
  )
  return new_name if new_name != "" else column_name

def refine_schema_names(schema):
  new_schema = {
    "column_names": [],
    "column_names_original": schema["column_names_original"],
    "column_types": schema["column_types"],
    "db_id": schema["db_id"],
    "foreign_keys": schema["foreign_keys"],
    "primary_keys": schema["primary_keys"],
    "table_names": [],
    "table_names_original": schema["table_names_original"],
  }
  for table in schema["table_names_original"]:
    corrected = _prompt_table(table)
    new_schema["table_names"].append(corrected)
  for col in schema["column_names_original"]:
    t_id = col[0]
    column_name = col[1]
    corrected = _prompt_column(column_name, new_schema["table_names"][t_id])
    new_schema["column_names"].append([t_id, corrected])
  return new_schema