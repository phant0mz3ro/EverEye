import sqlite3

# 1. Connect to the .db file (creates it if it doesn't exist)
conn = sqlite3.connect("detections.db")
cursor = conn.cursor()

'''
# 2. Create a table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT,
        age INTEGER
    )
""")'''

# 3. Insert data
#cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ("Alice", 28))
#conn.commit()  # Save changes

# 4. Read/Query data
cursor.execute("SELECT * FROM detections")
rows = cursor.fetchall()

for row in rows:
    print(row)

# 5. Close connection
conn.close()