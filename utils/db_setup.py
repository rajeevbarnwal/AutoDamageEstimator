import sqlite3

# 1. Connect (this creates the file if it doesn’t exist)
conn = sqlite3.connect("database/parts_costs.db")
c = conn.cursor()

# 2. Create the table (if it’s not already there)
c.execute("""
CREATE TABLE IF NOT EXISTS parts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  part_name TEXT UNIQUE,
  repair_cost REAL,
  replace_cost REAL
)
""")

# 3. Insert some sample parts & costs
#    You can expand this list later or import from CSV.
samples = [
  ("bumper", 150.0, 300.0),
  ("hood",    200.0, 400.0),
  ("door",    120.0, 250.0),
]
for name, repair, replace in samples:
    c.execute("""
      INSERT OR IGNORE INTO parts (part_name, repair_cost, replace_cost)
      VALUES (?, ?, ?)
    """, (name, repair, replace))

# 4. Save & close
conn.commit()
conn.close()

print("✔ parts_costs.db created and sample data inserted.")

