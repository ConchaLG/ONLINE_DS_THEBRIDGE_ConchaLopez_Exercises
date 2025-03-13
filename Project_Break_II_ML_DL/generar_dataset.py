import pandas as pd
import numpy as np
import random

# Crear un DataFrame con datos de ejemplo
def generar_datos_ejemplo(n):
    ticket_ids = [f'F{random.randint(1000, 9999)}' for _ in range(n)]
    ages = np.random.randint(18, 65, n)
    ticket_prices = np.round(np.random.uniform(10, 25, n), 2)
    movie_genres = np.random.choice(['Comedy', 'Drama', 'Horror', 'Action', 'Sci-Fi'], n)
    seat_types = np.random.choice(['Standard', 'VIP', 'Premium'], n)
    number_of_persons = np.random.choice([1, 2, 3, 4, 5, 6, 7], n)
    purchase_again = np.random.choice(['Yes', 'No'], n)
    years = np.random.choice(range(2020, 2025), n)
    
    return pd.DataFrame({
        'Ticket_ID': ticket_ids,
        'Age': ages,
        'Ticket_Price': ticket_prices,
        'Movie_Genre': movie_genres,
        'Seat_Type': seat_types,
        'Number_of_Person': number_of_persons,
        'Purchase_Again': purchase_again,
        'Year': years
    })

# Generar datos de ejemplo
datos_ejemplo = generar_datos_ejemplo(1000)

# Guardar el dataset de ejemplo
datos_ejemplo.to_csv('cinema_hall_ticket_sales_ampliado.csv', index=False)

print("El dataset de ejemplo ha sido guardado como 'cinema_hall_ticket_sales_ampliado.csv'")
