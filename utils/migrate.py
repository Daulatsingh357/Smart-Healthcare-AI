"""Tiny additive migration: adds nullable columns that exist in the models but not yet in the database.
Lets an existing database from the earlier project version upgrade without data loss."""
from sqlalchemy import inspect, text


def ensure_columns(db):
    insp = inspect(db.engine)
    added = []
    for table in db.metadata.sorted_tables:
        if not insp.has_table(table.name):
            continue
        have = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name in have or col.primary_key:
                continue
            ctype = col.type.compile(dialect=db.engine.dialect)
            default = ""
            if isinstance(col.default, type(None)) is False and getattr(col.default, "is_scalar", False):
                v = col.default.arg
                default = f" DEFAULT {int(v) if isinstance(v, bool) else repr(v)}"
            with db.engine.begin() as conn:
                conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {ctype}{default}'))
            added.append(f"{table.name}.{col.name}")
    return added
