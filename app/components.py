import pandas as pd
import streamlit as st


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Naredi sidebar filtre za poizvedbe"""
    st.sidebar.header('Filtri')

    date_range = st.sidebar.date_input(
        'Časovno obdobje:',
        value=(df['Date'].min().date(), df['Date'].max().date()),
        min_value=df['Date'].min().date(),
        max_value=df['Date'].max().date()
    )

    stores = st.sidebar.multiselect(
        'Trgovine',
        options=sorted(df['Store'].unique()),
        default=sorted(df['Store'].unique())
    )

    categories = st.sidebar.multiselect(
        'Kategorije:',
        options=sorted(df['Category'].unique()),
        default=sorted(df['Category'].unique())
    )

    price_range = st.sidebar.slider(
        "Cenovno območje (€):",
        min_value=float(df['Price'].min()),
        max_value=float(df['Price'].max()),
        value=(float(df['Price'].min()), float(df['Price'].max())),
        step=0.1
    )

    filtered_df = df.copy()

    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['Date'].dt.date >= date_range[0]) &
            (filtered_df['Date'].dt.date <= date_range[1])
        ]

    if stores:
        filtered_df = filtered_df[filtered_df['Store'].isin(stores)]

    if categories:
        filtered_df = filtered_df[filtered_df['Category'].isin(categories)]

    filtered_df = filtered_df[
        (filtered_df['Price'] >= price_range[0]) &
        (filtered_df['Price'] <= price_range[1])
    ]

    return filtered_df


def display_metrics(df: pd.DataFrame) -> None:
    """Pokaži najbolj pomembne informacije"""
    st.subheader("📊 Pregled")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Skupaj izdelkov", len(df))

    with col2:
        st.metric("Povprečna cena", f"{df['Price'].mean():.2f} €")

    with col3:
        st.metric("Najdražji", f"{df['Price'].max():.2f} €")

    with col4:
        st.metric("Najcenejši", f"{df['Price'].min():.2f} €")