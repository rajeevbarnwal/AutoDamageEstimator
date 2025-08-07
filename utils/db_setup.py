import sqlite3

conn = sqlite3.connect("database/parts_costs.db")
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS parts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  part_name TEXT UNIQUE,
  repair_cost REAL,
  replace_cost REAL
)
""")

samples = [
    ("bumper", 2222.0, 4444.0),
    ("hood", 3333.0, 6666.0),
    ("door", 2777.0, 5554.0),
    ("Quarter-panel", 1500.0, 3000.0),
    ("Front-wheel", 2800.0, 5600.0),
    ("Back-window", 4250.0, 9500.0),
    ("Trunk", 1800.0, 3600.0),
    ("Front-door", 2200.0, 4400.0),
    ("Rocker-panel", 1300.0, 2600.0),
    ("Grille", 2000.0, 4000.0),
    ("Windshield", 5000.0, 1000.0),
    ("Front-window", 2500.0, 5000.0),
    ("Back-door", 2000.0, 4000.0),
    ("Headlight", 2500.0, 5000.0),
    ("Back-wheel", 4000.0, 8000.0),
    ("Back-windshield", 5000.0, 1000.0),
    ("Hood", 4500.0, 9000.0),
    ("Fender", 1450.0, 2900.0),
    ("Tail-light", 2500.0, 5000.0),
    ("License-plate", 1050.0, 2100.0),
    ("Front-bumper", 3050.0, 7000.0),
    ("Back-bumper", 3050.0, 6100.0),
    ("Mirror", 2000.0, 10000.0),
    ("Roof", 4000.0, 8000.0),
    ("Missing part", 2000.0, 4000.0),  # Damage costs
    ("Broken part", 2500.0, 5000.0),
    ("Scratch", 3050.0, 6100.0),
    ("Cracked", 2500.0, 5000.0),
    ("Dent", 4000.0, 8000.0),
    ("Flaking", 3150.0, 6300.0),
    ("Paint chip", 2500.0, 5000.0),
    ("Corrosion", 2500.0, 5000.0),
]
for name, repair, replace in samples:
    c.execute("""
      INSERT OR REPLACE INTO parts (part_name, repair_cost, replace_cost)
      VALUES (?, ?, ?)
    """, (name, repair, replace))

conn.commit()
conn.close()

print("✔ parts_costs.db created and sample data inserted.")
