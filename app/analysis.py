import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from matplotlib_venn import venn2, venn3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_top_bottom_prices_over_time(df: pd.DataFrame):
    st.subheader("🔝🔻 Najvišje in najnižje cene skozi čas (mesečno)")
    if df.empty or 'Date' not in df.columns or 'Price' not in df.columns:
        st.warning("Podatki niso na voljo ali manjkajo potrebni stolpci.")
        return
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    monthly = df.set_index('Date').resample('M')['Price']
    max_monthly = monthly.max()
    min_monthly = monthly.min()
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(max_monthly.index, max_monthly.values, color='red', label='Najvišja cena')
    ax.plot(min_monthly.index, min_monthly.values, color='green', label='Najnižja cena')
    ax.set_title('Najvišja in najnižja cena izdelka skozi čas (mesečno)')
    ax.set_xlabel('Mesec')
    ax.set_ylabel('Cena (€)')
    ax.legend()
    st.pyplot(fig)

def plot_extreme_items_halfyear(df: pd.DataFrame):
    st.subheader("🏆 Najdražji in najcenejši izdelek po polletjih")
    if df.empty or 'Date' not in df.columns or 'Price' not in df.columns or 'Product Name Grouped' not in df.columns:
        st.warning("Podatki niso na voljo ali manjkajo potrebni stolpci.")
        return
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['YearHalf'] = df['Date'].dt.year.astype(str) + '-H' + ((df['Date'].dt.month - 1) // 6 + 1).astype(str)
    idx_max = df.groupby('YearHalf')['Price'].idxmax()
    idx_min = df.groupby('YearHalf')['Price'].idxmin()
    most_expensive = df.loc[idx_max, ['YearHalf', 'Product Name Grouped', 'Price']]
    cheapest = df.loc[idx_min, ['YearHalf', 'Product Name Grouped', 'Price']]
    halfyear_extremes = pd.merge(
        most_expensive, 
        cheapest, 
        on='YearHalf', 
        suffixes=('_Max', '_Min')
    )
    st.dataframe(halfyear_extremes.rename(columns={
        'Product Name Grouped_Max': 'Najdražji izdelek',
        'Price_Max': 'Cena najdražjega',
        'Product Name Grouped_Min': 'Najcenejši izdelek',
        'Price_Min': 'Cena najcenejšega'
    }), use_container_width=True)

    # Najpogostejši najdražji/najcenejši izdelek po polletjih
    st.subheader("🏅 Najpogosteje najdražji in najcenejši izdelek po polletjih")
    top_counts = most_expensive['Product Name Grouped'].value_counts().head(10)
    bottom_counts = cheapest['Product Name Grouped'].value_counts().head(10)
    fig, axs = plt.subplots(1, 2, figsize=(14,4))
    axs[0].barh(top_counts.index[::-1], top_counts.values[::-1], color='red')
    axs[0].set_title('Top 10 najdražjih izdelkov')
    axs[1].barh(bottom_counts.index[::-1], bottom_counts.values[::-1], color='green')
    axs[1].set_title('Top 10 najcenejših izdelkov')
    plt.tight_layout()
    st.pyplot(fig)

def plot_biggest_price_changes(df: pd.DataFrame):
    st.subheader("📈 Izdelki z največjimi spremembami cene skozi čas")
    if df.empty or 'Date' not in df.columns or 'Price' not in df.columns or 'Product Name Grouped' not in df.columns:
        st.warning("Podatki niso na voljo ali manjkajo potrebni stolpci.")
        return
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    price_changes = []
    for name, group in df.sort_values('Date').groupby('Product Name Grouped'):
        changes = group['Price'].diff().dropna()
        if not changes.empty:
            price_changes.append({
                'Product': name,
                'Avg Change': changes.abs().mean(),
                'Num Changes': changes.count()
            })
    changes_df = pd.DataFrame(price_changes)
    changes_df = changes_df[changes_df['Num Changes'] >= 5].sort_values('Avg Change', ascending=False)
    st.write("Top 15 izdelkov z največjo povprečno spremembo cene:")
    st.dataframe(changes_df.head(15), use_container_width=True)

def plot_price_segmentation(df: pd.DataFrame):
    st.subheader("📊 Segmentacija cen skozi čas")
    if df.empty or 'Price' not in df.columns or 'Date' not in df.columns:
        st.warning("Podatki niso na voljo ali manjkajo potrebni stolpci.")
        return
    terciles = df['Price'].quantile([0.33, 0.66])
    def price_segment(price):
        if price <= terciles.iloc[0]:
            return 'Nizka'
        elif price <= terciles.iloc[1]:
            return 'Srednja'
        else:
            return 'Visoka'
    df['Price Segment'] = df['Price'].apply(price_segment)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    segment_trend = df.groupby([df['Date'].dt.to_period('M'), 'Price Segment']).size().unstack().fillna(0)
    segment_trend = segment_trend.div(segment_trend.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(12,5))
    segment_trend.plot.area(stacked=True, color=['green','gold','red'], ax=ax)
    ax.set_title('Delež cenovnih segmentov skozi čas')
    ax.set_xlabel('Mesec')
    ax.set_ylabel('Delež')
    ax.legend(title='Segment')
    st.pyplot(fig)

def inflation_analysis_section():
    st.subheader("💸 Inflacija: primerjava nominalnih in realnih cen")

    import os
    import matplotlib.pyplot as plt

    # 1. Load and parse inflation index
    inflation_path = os.path.join(os.path.dirname(__file__), '..', 'razmerje_cen.csv')
    data = []
    try:
        with open(inflation_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('('):
                    if '\t' in line:
                        parts = line.split('\t', 1)
                    elif ' ' in line and line.count(' ') == 1:
                        parts = line.split(' ', 1)
                    else:
                        import re
                        match = re.match(r'([^\d,]+)\s*([0-9,]+)', line)
                        if match:
                            parts = [match.group(1).strip(), match.group(2).strip()]
                        else:
                            continue
                    if len(parts) == 2:
                        date_str = parts[0].strip()
                        price_str = parts[1].strip().replace(',', '.')
                        try:
                            price = float(price_str)
                            data.append([date_str, price])
                        except Exception:
                            continue
        inflation_df = pd.DataFrame(data, columns=['Date_Month', 'Inflation_Index'])
        if inflation_df.empty:
            st.warning("Inflacijski podatki niso na voljo.")
            return
    except Exception as e:
        st.warning(f"Napaka pri branju inflacijskih podatkov: {e}")
        return

    # 2. Clean and format inflation data
    inflation_df['Date_Month_Clean'] = inflation_df['Date_Month'].str.replace('M', '-') + '-01'
    inflation_df['Date_Month_Clean'] = pd.to_datetime(inflation_df['Date_Month_Clean'], errors='coerce')
    inflation_df['Year'] = inflation_df['Date_Month_Clean'].dt.year
    inflation_df['Month'] = inflation_df['Date_Month_Clean'].dt.month
    inflation_df = inflation_df.dropna()

    # 3. Show inflation index trend
    st.write("**Inflacijski indeks skozi čas:**")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(inflation_df['Date_Month_Clean'], inflation_df['Inflation_Index'], marker='o')
    ax.set_title('Inflacijski indeks (SURS)')
    ax.set_xlabel('Mesec')
    ax.set_ylabel('Indeks (2015=100)')
    st.pyplot(fig)

    # 4. Load your main data (top stores, grouped, no outliers)
    price_path = os.path.join(os.path.dirname(__file__), '..', 'vsikatalogi_ready_no_outliers_topstores_grouped.csv')
    try:
        df = pd.read_csv(price_path, sep=';')
    except Exception as e:
        st.warning(f"Napaka pri branju glavnih podatkov: {e}")
        return

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # 5. Merge with inflation index
    df = df.merge(
        inflation_df[['Year', 'Month', 'Inflation_Index']],
        on=['Year', 'Month'],
        how='left'
    )
    df['Real_Price'] = df['Price (€)'] * (100 / df['Inflation_Index'])

    # 6. Compare nominal vs real prices per store (for products in >=2 stores)
    product_store_counts = df.groupby('Product Name Grouped')['Store'].nunique()
    common_products = product_store_counts[product_store_counts >= 2].index
    df_common = df[df['Product Name Grouped'].isin(common_products)].copy()

    pivot_real = df_common.pivot_table(index='Product Name Grouped', columns='Store', values='Real_Price', aggfunc='mean')
    pivot_nominal = df_common.pivot_table(index='Product Name Grouped', columns='Store', values='Price (€)', aggfunc='mean')

    # 7. Cheapest store counts
    cheapest_store_real = pivot_real.idxmin(axis=1)
    cheapest_count_real = cheapest_store_real.value_counts()
    cheapest_store_nominal = pivot_nominal.idxmin(axis=1)
    cheapest_count_nominal = cheapest_store_nominal.value_counts()

    # 8. Bar plots: share of cheapest products per store
    st.write("**Delež najcenejših izdelkov po trgovinah (nominalne in realne cene):**")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    (cheapest_count_nominal / len(pivot_nominal) * 100).plot(kind='bar', ax=ax1, color='lightcoral')
    ax1.set_title('Nominalne cene')
    ax1.set_ylabel('Delež (%)')
    ax1.tick_params(axis='x', rotation=45)
    (cheapest_count_real / len(pivot_real) * 100).plot(kind='bar', ax=ax2, color='lightseagreen')
    ax2.set_title('Realne cene (prilagojene inflaciji)')
    ax2.set_ylabel('Delež (%)')
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # 9. Bar plots: average price per store
    st.write("**Povprečna cena po trgovinah (nominalno vs realno):**")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    pivot_nominal.mean().sort_values().plot(kind='bar', ax=ax1, color='orange')
    ax1.set_title('Povprečna nominalna cena')
    ax1.set_ylabel('Cena (€)')
    ax1.tick_params(axis='x', rotation=45)
    pivot_real.mean().sort_values().plot(kind='bar', ax=ax2, color='steelblue')
    ax2.set_title('Povprečna realna cena (prilagojena inflaciji)')
    ax2.set_ylabel('Realna cena (€)')
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # 10. Text summary
    st.info(
        "Primerjava prikazuje, kako se razlike med trgovinami spremenijo, ko upoštevamo inflacijo. "
        "Realne cene omogočajo bolj pošteno primerjavo skozi čas."
    )





def time_analysis(df: pd.DataFrame) -> None:
    """Časovna analiza s cenami po kategorijah"""

    if df.empty:
        st.warning("Ni podatkov za analizo")
        return

    # 1. Povprečna cena skozi čas (prvi graf)
    plot_average_price_trend(df)

    # 2. Sezonska analiza
    st.subheader("🗓️ Sezonska analiza")
    df_seasonal = df.copy()
    df_seasonal['Month'] = df_seasonal['Date'].dt.month
    df_seasonal['Month_Name'] = df_seasonal['Date'].dt.strftime('%B')
    df_seasonal['DayOfWeek_Name'] = df_seasonal['Date'].dt.strftime('%A')

    top_categories = df['Category'].value_counts().head(8).index.tolist()

    col1, col2 = st.columns(2)
    with col1:
        # Mesečni trendi cen za top kategorije
        monthly_category = df_seasonal.groupby(['Month_Name', 'Category'])['Price'].mean().reset_index()
        monthly_top = monthly_category[monthly_category['Category'].isin(top_categories[:5])]
        if not monthly_top.empty:
            fig_monthly = px.line(monthly_top, x='Month_Name', y='Price', color='Category',
                                  title='Mesečni trendi cen (top 5 kategorij)',
                                  labels={'Price': 'Povprečna cena (€)', 'Month_Name': 'Mesec'})
            st.plotly_chart(fig_monthly, use_container_width=True)
    with col2:
        # Tedenski vzorci
        weekly_pattern = df_seasonal.groupby('DayOfWeek_Name')['Price'].mean().reset_index()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_pattern['DayOfWeek_Name'] = pd.Categorical(weekly_pattern['DayOfWeek_Name'],
                                                          categories=day_order, ordered=True)
        weekly_pattern = weekly_pattern.sort_values('DayOfWeek_Name')
        fig_weekly = px.bar(weekly_pattern, x='DayOfWeek_Name', y='Price',
                            title='Povprečne cene po dnevih v tednu',
                            labels={'Price': 'Povprečna cena (€)', 'DayOfWeek_Name': 'Dan'})
        st.plotly_chart(fig_weekly, use_container_width=True)

    # 3. Analiza cenovnih razponov
    st.subheader("📏 Analiza cenovnih razponov")
    price_ranges = []
    for category in top_categories:
        cat_data = df[df['Category'] == category]['Price']
        if len(cat_data) > 0:
            q1 = cat_data.quantile(0.25)
            q3 = cat_data.quantile(0.75)
            price_ranges.append({
                'Kategorija': category,
                'Q1': q1,
                'Mediana': cat_data.median(),
                'Q3': q3,
                'IQR': q3 - q1,
                'Min': cat_data.min(),
                'Max': cat_data.max()
            })
    if price_ranges:
        fig_box = go.Figure()
        for category in top_categories:
            cat_prices = df[df['Category'] == category]['Price']
            fig_box.add_trace(go.Box(
                y=cat_prices,
                name=category,
                boxpoints='outliers'
            ))
        fig_box.update_layout(
            title='Porazdelitev cen po kategorijah (škatlasti diagram)',
            yaxis_title='Cena (€)',
            xaxis_title='Kategorija'
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # 4. Ostale analize (kot prej)
    st.subheader("📈 Časovni trendi cen")

    # Možnosti analize
    show_relative = st.checkbox('Prikaži relativne spremembe', value=False,
                                help='Prikaži spremembe v % glede na prvo vrednost')

    # Trendi cen kategorij skozi čas
    st.subheader("💰 Spremembe cen po kategorijah")
    daily_category_prices = df.groupby(['Date', 'Category'])['Price'].mean().reset_index()
    filtered_data = daily_category_prices[daily_category_prices['Category'].isin(top_categories)]

    if show_relative:
        for category in top_categories:
            category_data = filtered_data[filtered_data['Category'] == category].copy()
            if not category_data.empty:
                first_price = category_data['Price'].iloc[0]
                relative_change = ((category_data['Price'] - first_price) / first_price * 100)
                filtered_data.loc[filtered_data['Category'] == category, 'Price'] = relative_change
        y_label = 'Sprememba cene (%)'
        title = 'Relativne spremembe cen po kategorijah'
    else:
        y_label = 'Povprečna cena (€)'
        title = 'Povprečne cene po kategorijah skozi čas'

    fig = px.line(filtered_data, x='Date', y='Price', color='Category',
                  title=title,
                  labels={'Price': y_label, 'Date': 'Datum', 'Category': 'Kategorija'})
    fig.update_layout(height=500, legend=dict(orientation="v", x=1.02, y=1))
    st.plotly_chart(fig, use_container_width=True)

    # Analiza volatilnosti cen
    st.subheader("📊 Volatilnost cen po kategorijah")
    volatility_data = []
    for category in top_categories:
        category_prices = df[df['Category'] == category]['Price']
        if len(category_prices) > 1:
            cv = category_prices.std() / category_prices.mean() * 100
            volatility_data.append({
                'Kategorija': category,
                'Volatilnost (%)': cv,
                'Povprečna cena': category_prices.mean(),
                'Min cena': category_prices.min(),
                'Max cena': category_prices.max(),
                'Število izdelkov': len(category_prices)
            })
    if volatility_data:
        volatility_df = pd.DataFrame(volatility_data).sort_values('Volatilnost (%)', ascending=False)
        col1, col2 = st.columns(2)
        with col1:
            fig_vol = px.bar(volatility_df, x='Kategorija', y='Volatilnost (%)',
                             title='Cenovna volatilnost po kategorijah',
                             color='Volatilnost (%)', color_continuous_scale='Reds')
            fig_vol.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_vol, use_container_width=True)
        with col2:
            fig_scatter = px.scatter(volatility_df, x='Povprečna cena', y='Volatilnost (%)',
                                     size='Število izdelkov', hover_name='Kategorija',
                                     title='Volatilnost vs Povprečna cena',
                                     labels={'Povprečna cena': 'Povprečna cena (€)'})
            st.plotly_chart(fig_scatter, use_container_width=True)
        st.write("**Podrobnosti volatilnosti:**")
        st.dataframe(volatility_df.round(2), use_container_width=True)

    # Povzetek ugotovitev
    st.subheader("🔍 Ključne ugotovitve")
    if volatility_data:
        most_volatile = volatility_df.iloc[0]['Kategorija']
        least_volatile = volatility_df.iloc[-1]['Kategorija']
        date_range = df['Date'].max() - df['Date'].min()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Najbolj volatilna kategorija:**\n{most_volatile}")
        with col2:
            st.success(f"**Najbolj stabilna kategorija:**\n{least_volatile}")
        with col3:
            st.warning(f"**Analizirano obdobje:**\n{date_range.days} dni")
        expensive_categories = volatility_df.nlargest(3, 'Povprečna cena')['Kategorija'].tolist()
        cheap_categories = volatility_df.nsmallest(3, 'Povprečna cena')['Kategorija'].tolist()
        st.write("**Najdražje kategorije:**", ", ".join(expensive_categories))
        st.write("**Najcenejše kategorije:**", ", ".join(cheap_categories))

    # Dodatne časovne analize
    plot_top_bottom_prices_over_time(df)
    plot_extreme_items_halfyear(df)
    plot_biggest_price_changes(df)
    plot_price_segmentation(df)
    inflation_analysis_section()


def plot_average_price_trend(df: pd.DataFrame):
    st.subheader("📈 Povprečna cena skozi čas")
    if df.empty or 'Date' not in df.columns or 'Price' not in df.columns:
        st.warning("Podatki niso na voljo ali manjkajo potrebni stolpci.")
        return

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['YearMonth'] = df['Date'].dt.to_period('M')
    avg_price = df.groupby('YearMonth')['Price'].mean().reset_index()
    avg_price['YearMonth'] = avg_price['YearMonth'].astype(str)

    fig = px.line(avg_price, x='YearMonth', y='Price',
                  title='Povprečna cena izdelkov skozi čas (mesečno)',
                  labels={'YearMonth': 'Mesec', 'Price': 'Povprečna cena (€)'})
    fig.update_layout(xaxis_tickangle=45, height=400)
    st.plotly_chart(fig, use_container_width=True)


def plot_store_venn(df: pd.DataFrame):
    st.subheader("🔗 Prekrivanje izdelkov med trgovinami (Venn diagram)")

    if df.empty or 'Store' not in df.columns or 'Product Name Grouped' not in df.columns:
        st.warning("Podatki niso na voljo ali manjkajo potrebni stolpci.")
        return

    stores = sorted(df['Store'].dropna().unique())
    if len(stores) < 2:
        st.info("Za Venn diagram potrebujemo vsaj dve trgovini.")
        return

    # Select stores for comparison
    selected_stores = st.multiselect(
        "Izberi 2 ali 3 trgovine za primerjavo:",
        stores,
        default=stores[:3] if len(stores) >= 3 else stores[:2]
    )

    if len(selected_stores) < 2 or len(selected_stores) > 3:
        st.info("Izberi 2 ali 3 trgovine.")
        return

    # Prepare sets
    sets = [set(df[df['Store'] == store]['Product Name Grouped']) for store in selected_stores]

    fig, ax = plt.subplots(figsize=(6, 5))
    if len(selected_stores) == 2:
        venn2(sets, set_labels=selected_stores, ax=ax)
    elif len(selected_stores) == 3:
        venn3(sets, set_labels=selected_stores, ax=ax)
    st.pyplot(fig)

def plot_store_price_trends(df: pd.DataFrame):
    st.subheader("🏪 Povprečna cena po trgovinah skozi čas")

    if df.empty or 'Store' not in df.columns or 'Date' not in df.columns or 'Price' not in df.columns:
        st.warning("Podatki niso na voljo ali manjkajo potrebni stolpci.")
        return

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Year'] = df['Date'].dt.year

    # Pivot: average price per store per year
    pivot = df.pivot_table(index='Year', columns='Store', values='Price', aggfunc='mean')
    pivot = pivot.sort_index()

    fig = px.line(
        pivot,
        x=pivot.index,
        y=pivot.columns,
        labels={'value': 'Povprečna cena (€)', 'Year': 'Leto', 'variable': 'Trgovina'},
        title='Povprečna cena po trgovinah skozi leta'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def store_analysis(df: pd.DataFrame) -> None:
    """Analiza po trgovinah"""
    st.subheader('🏪 Analiza trgovin')

    if df.empty:
        st.write('Ni podatkov za analizo')
        return

    # Statistike trgovin
    store_stats = df.groupby('Store').agg({
        'Price': ['mean', 'median', 'std', 'count']
    }).round(2)
    store_stats.columns = ['Povprečje', 'Mediana', 'Std', 'Število']
    store_stats = store_stats.reset_index()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(store_stats, x='Store', y='Povprečje', title='Povprečne cene po trgovinah')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df, x='Store', y='Price', title='Porazdelitev cen')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Statistika trgovin')
    st.dataframe(store_stats)


def category_analysis(df: pd.DataFrame) -> None:
    """Analiza kategorij"""
    st.subheader("📦 Analiza kategorij")

    if df.empty:
        st.warning("Ni podatkov za analizo")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Porazdelitev kategorij
        category_counts = df['Category'].value_counts()
        fig = px.pie(values=category_counts.values, names=category_counts.index,
                     title='Porazdelitev po kategorijah')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Povprečne cene po kategorijah
        category_prices = df.groupby('Category')['Price'].mean().sort_values(ascending=False)
        fig = px.bar(x=category_prices.index, y=category_prices.values,
                     title='Povprečne cene po kategorijah')
        st.plotly_chart(fig, use_container_width=True)

    # Analiza kategorij vs trgovin
    if len(df) > 0:
        category_store = df.groupby(['Category', 'Store'])['Price'].mean().unstack(fill_value=0)
        if not category_store.empty:
            fig = px.imshow(category_store.values,
                            x=category_store.columns,
                            y=category_store.index,
                            title='Cene: Kategorije vs Trgovine')
            st.plotly_chart(fig, use_container_width=True)


def data_mining_analysis(df: pd.DataFrame) -> None:
    """Analiza podatkovnega rudarjenja"""
    st.subheader("🔍 Data Mining")

    if df.empty:
        st.warning("Ni podatkov za analizo")
        return

    # K-means grupiranje
    st.subheader("K-means grupiranje")

    if len(df) <= 10:
        st.warning("Premalo podatkov za grupiranje (potrebnih več kot 10)")
        return

    # Ponastavi indekse, da se izognemo težavam z indeksiranjem
    df_reset = df.reset_index(drop=True)

    # Pripravi značilnosti
    feature_data = pd.DataFrame({
        'Price': df_reset['Price'],
        'Price_log': np.log1p(df_reset['Price']),
        'Name_length': df_reset['Product Name'].str.len(),
        'Days_since_start': (df_reset['Date'] - df_reset['Date'].min()).dt.days
    })

    # Obravnavaj manjkajoče vrednosti
    feature_data = feature_data.dropna()

    if len(feature_data) <= 5:
        st.warning("Premalo podatkov za grupiranje (potrebnih več kot 5)")
        return

    # Skaliraj značilnosti
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_data)

    # Grupiranje
    n_clusters = st.slider("Število skupin:", 2, 6, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    col1, col2 = st.columns(2)

    with col1:
        # PCA vizualizacija
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(scaled_features)

        fig = px.scatter(x=pca_features[:, 0], y=pca_features[:, 1],
                         color=clusters, title='Vizualizacija')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Uporabi ponastavljeni dataframe z ustreznimi indeksi
        cluster_df = df_reset.iloc[feature_data.index].copy()
        cluster_df['Cluster'] = clusters

        cluster_stats = cluster_df.groupby('Cluster').agg({
            'Price': ['mean', 'count']
        }).round(2)

        cluster_stats.columns = ['Povprečna cena', 'Število']
        st.write("**Statistike skupin:**")
        st.dataframe(cluster_stats)

    # Korelacijska analiza
    st.subheader("Korelacijska analiza")

    if len(df) <= 1:
        st.warning("Premalo podatkov za korelacijsko analizo")
        return

    # Uporabi isti ponastavljeni dataframe
    df_reset = df.reset_index(drop=True)

    corr_data = pd.DataFrame({
        'Price': df_reset['Price'],
        'Name_length': df_reset['Product Name'].str.len(),
        'Days_from_start': (df_reset['Date'] - df_reset['Date'].min()).dt.days,
        'Is_weekend': df_reset['Date'].dt.weekday >= 5
    })

    correlation_matrix = corr_data.corr()
    fig = px.imshow(correlation_matrix, title='Korelacijska matrika')
    st.plotly_chart(fig, use_container_width=True)

    # Prikaži korelacijske vrednosti
    st.write("**Korelacije s ceno:**")
    price_corr = correlation_matrix['Price'].sort_values(ascending=False)
    for var, corr in price_corr.items():
        if var != 'Price':
            st.write(f"- {var}: {corr:.3f}")
