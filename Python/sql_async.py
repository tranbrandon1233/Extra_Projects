from datetime import datetime
from sqlalchemy import String, Integer, DateTime, Float, select
from sqlalchemy.orm import DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import asyncio


class Base(DeclarativeBase):
    pass

class Product(Base):
    __tablename__ = "products"
    
    id = mapped_column(Integer, primary_key=True)
    name = mapped_column(String(100), nullable=False)
    price = mapped_column(Float, nullable=False)
    stock = mapped_column(Integer, default=0)
    created_at = mapped_column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"Product(id={self.id}, name='{self.name}', price={self.price}, stock={self.stock})"

# Create async engine
engine = create_async_engine(
    "sqlite+aiosqlite:///test.db",
    echo=True
)

# Function to create test data
async def create_test_data():
    async with AsyncSession(engine) as session:
        async with session.begin():
            # Create some test products
            test_products = [
                Product(
                    name="Laptop",
                    price=999.99,
                    stock=50
                ),
                Product(
                    name="Smartphone",
                    price=599.99,
                    stock=100
                ),
                Product(
                    name="Headphones",
                    price=199.99,
                    stock=75
                ),
                Product(
                    name="Tablet",
                    price=399.99,
                    stock=30
                )
            ]
            session.add_all(test_products)

async def main():
    async with engine.begin() as conn:
        await conn.run_sync(Product.metadata.create_all)  # Create tables
    
    # Create test data
    await create_test_data()
    
    # Query the data
    async with AsyncSession(engine) as session:
        async with session.begin():
            result = await session.execute(select(Product))
            print("\n\nPrinting products:")
            for row in result.scalars():
                print(row)

if __name__ == "__main__":
    asyncio.run(main())