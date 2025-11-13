import asyncio
from twikit import Client
import pandas as pd


async def main():
    # Crea el cliente y haz login
    client = Client("en-US")
    await client.login(
        auth_info_1="kericode01",  # tu usuario o correo
        auth_info_2="kericode01",  # puede ser el mismo
        password="Orange:3//01",  # tu contraseña
    )

    # Busca tweets
    tweets = await client.search_tweet("AI", product="Latest", count=50)
    print(f"Número de tweets obtenidos: {len(tweets)}")

    # Convierte a DataFrame
    data = [{"user": t.user.name, "text": t.text, "date": t.created_at} for t in tweets]
    df = pd.DataFrame(data)
    df.to_csv("tweets_twikit.csv", index=False)
    print(df.head())


# Ejecutar
if __name__ == "__main__":
    asyncio.run(main())
