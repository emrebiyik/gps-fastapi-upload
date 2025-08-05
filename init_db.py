from database import engine, Base
from models import ImageGPSData

def init():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init()