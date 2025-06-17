from sqlalchemy import create_engine, MetaData, text

# Connection config
db_user = "root"
db_password = "Test1234"
db_host = "77.93.155.81"
db_name = "trafficOptimization"

connection_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"

# Create engine
engine = create_engine(
    connection_url,
    pool_recycle=280,
    pool_pre_ping=True
)

# Drop all tables
def drop_all_tables():
    try:
        with engine.connect() as conn:
            print("Connected. Disabling foreign key checks...")
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))

            meta = MetaData()
            meta.reflect(bind=engine)

            print(f"Dropping {len(meta.tables)} tables...")
            for table in meta.sorted_tables:
                print(f"Dropping table {table.name}...")
                conn.execute(text(f"DROP TABLE IF EXISTS `{table.name}`"))

            print("Re-enabling foreign key checks...")
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

            print("✅ All tables dropped successfully.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    drop_all_tables()
