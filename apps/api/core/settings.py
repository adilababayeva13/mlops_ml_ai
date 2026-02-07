from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+psycopg2://app:apppass@localhost:5432/what_to_eat"
    JWT_SECRET: str = "CHANGE_ME_IN_PROD"
    JWT_ALG: str = "HS256"
    JWT_EXPIRES_MIN: int = 60

settings = Settings()
