import pandas as pd
import streamlit as st

from analysis import time_analysis, store_analysis, category_analysis, data_mining_analysis
from components import sidebar_filters, display_metrics
from data_loader import load_and_process_data

def main() -> None:
    # Config
    st.set_page_config(
        page_title='PR Projekt 2025',
        layout='wide'
    )

    st.title('PR Projekt 2025')
    st.subheader('Analiza cen v trgovinah')

    # Naloži podatke
    df = load_and_process_data()

    if df is not None:
        filtered_df = sidebar_filters(df)
        display_metrics(filtered_df)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Časovni trendi",
            "🏪 Analiza trgovin",
            "📦 Kategorije",
            "🔍 Data Mining",
            "📋 Podatki"
        ])

        with tab1:
            time_analysis(filtered_df)

        with tab2:
            store_analysis(filtered_df)

        with tab3:
            category_analysis(filtered_df)

        with tab4:
            data_mining_analysis(filtered_df)

        with tab5:
            display_raw_data_tab(filtered_df)


def display_raw_data_tab(df: pd.DataFrame) -> None:
    """Prikaži surove podatke"""
    st.subheader('Surovi podatki')

    st.write("**Stolpci podatkov:**")
    st.write("- **Product Name**: Originalno ime izdelka")
    st.write("- **Product Name Grouped**: Ime skupine podobnih izdelkov")
    st.write("- **Is_Grouped**: Ali je izdelek del skupine (True/False)")

    search = st.text_input('Iskanje po imenu izdelka:')
    if search:
        mask = (df['Product Name'].str.contains(search, case=False, na=False) |
                df['Product Name Grouped'].str.contains(search, case=False, na=False))
        display_df = df[mask]
    else:
        display_df = df

    st.write(f'Prikazanih izpisov: {len(display_df)}')
    st.dataframe(display_df, use_container_width=True)

    csv = display_df.to_csv(index=False)
    st.download_button('Prenesi CSV', csv, 'filtered_data.csv', 'text/csv')


if __name__ == '__main__':
    main()
