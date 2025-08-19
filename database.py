import sqlite3
import os

DB_FILE = "data/sales.db"
DB_DIR = "data"
REPORTS_DIR = "reports"

def setup_project():
    """Creates the database, data directory, and reports directory."""
    # Ensure the data and reports directories exist
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)

    # Remove the old database file if it exists
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        print("Creating tables...")
        # Create sales table
        cursor.execute('''
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT NOT NULL,
            quantity_sold INTEGER NOT NULL,
            price_per_unit REAL NOT NULL,
            sale_date TEXT NOT NULL
        )
        ''')

        
        cursor.execute('''
        CREATE TABLE employee_salaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            ssn TEXT NOT NULL,
            salary REAL NOT NULL
        )
        ''')

        print("Inserting sample data...")
        # Sample sales data for Q2 2025
        sales_data = [
            ('Solar Panel Pro', 150, 250.00, '2025-04-15'),
            ('Inverter X1', 200, 150.50, '2025-04-20'),
            ('Mounting Bracket', 500, 25.75, '2025-05-05'),
            ('Solar Panel Pro', 120, 255.00, '2025-05-10'),
            ('Battery Pack 5kWh', 80, 1200.00, '2025-06-01'),
            ('Inverter X1', 180, 152.00, '2025-06-15'),
            ('Solar Panel Pro', 200, 260.00, '2025-06-22')
        ]
        cursor.executemany('INSERT INTO sales (product_name, quantity_sold, price_per_unit, sale_date) VALUES (?, ?, ?, ?)', sales_data)

        # Sample sensitive data for the attack demo
        salaries_data = [
            ('Alice Johnson', '123-456-7890', 95000.00),
            ('Bob Williams', '098-765-4321', 115000.00)
        ]
        cursor.executemany('INSERT INTO employee_salaries (name, ssn, salary) VALUES (?, ?, ?)', salaries_data)


        conn.commit()
        print(f"Database '{DB_FILE}' created and populated successfully.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    setup_project()
