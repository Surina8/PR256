import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def time_analysis(df: pd.DataFrame) -> None:
    """Časovna analiza s cenami po kategorijah"""
    st.subheader("📈 Časovni trendi cen")

    if df.empty:
        st.warning("Ni podatkov za analizo")
        return

    # Možnosti analize
    show_relative = st.checkbox('Prikaži relativne spremembe', value=False,
                                help='Prikaži spremembe v % glede na prvo vrednost')

    # Trendi cen kategorij skozi čas
    st.subheader("💰 Spremembe cen po kategorijah")

    # Povprečne dnevne cene po kategorijah
    daily_category_prices = df.groupby(['Date', 'Category'])['Price'].mean().reset_index()

    # Top kategorije po številu izdelkov
    top_categories = df['Category'].value_counts().head(8).index.tolist()
    filtered_data = daily_category_prices[daily_category_prices['Category'].isin(top_categories)]

    if show_relative:
        # Relativne spremembe (% od prve vrednosti)
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

    # Glavni graf trendov cen
    fig = px.line(filtered_data, x='Date', y='Price', color='Category',
                  title=title,
                  labels={'Price': y_label, 'Date': 'Datum', 'Category': 'Kategorija'})

    fig.update_layout(height=500, legend=dict(orientation="v", x=1.02, y=1))
    st.plotly_chart(fig, use_container_width=True)

    # Analiza volatilnosti cen
    st.subheader("📊 Volatilnost cen po kategorijah")

    # Izračunaj volatilnost cen (koeficient variacije)
    volatility_data = []
    for category in top_categories:
        category_prices = df[df['Category'] == category]['Price']
        if len(category_prices) > 1:
            cv = category_prices.std() / category_prices.mean() * 100  # Koeficient variacije
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
            # Graf volatilnosti
            fig_vol = px.bar(volatility_df, x='Kategorija', y='Volatilnost (%)',
                             title='Cenovna volatilnost po kategorijah',
                             color='Volatilnost (%)', color_continuous_scale='Reds')
            fig_vol.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_vol, use_container_width=True)

        with col2:
            # Razpršeni graf volatilnost vs povprečna cena
            fig_scatter = px.scatter(volatility_df, x='Povprečna cena', y='Volatilnost (%)',
                                     size='Število izdelkov', hover_name='Kategorija',
                                     title='Volatilnost vs Povprečna cena',
                                     labels={'Povprečna cena': 'Povprečna cena (€)'})
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Tabela volatilnosti
        st.write("**Podrobnosti volatilnosti:**")
        st.dataframe(volatility_df.round(2), use_container_width=True)

    # Sezonska analiza
    st.subheader("🗓️ Sezonska analiza")

    # Dodaj mesec in dan v tednu za sezonsko analizo
    df_seasonal = df.copy()
    df_seasonal['Month'] = df_seasonal['Date'].dt.month
    df_seasonal['Month_Name'] = df_seasonal['Date'].dt.strftime('%B')
    df_seasonal['DayOfWeek_Name'] = df_seasonal['Date'].dt.strftime('%A')

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

    # Analiza cenovnih razponov
    st.subheader("📏 Analiza cenovnih razponov")

    # Cenovne razpone za vsako kategorijo
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
        # Ustvari škatlasti diagram
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

    # Povzetek ugotovitev
    st.subheader("🔍 Ključne ugotovitve")

    # Izračunaj zanimive vpoglede
    if not volatility_df.empty:
        most_volatile = volatility_df.iloc[0]['Kategorija']
        least_volatile = volatility_df.iloc[-1]['Kategorija']

        # Spremembe cen skozi čas
        date_range = df['Date'].max() - df['Date'].min()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.info(f"**Najbolj volatilna kategorija:**\n{most_volatile}")

        with col2:
            st.success(f"**Najbolj stabilna kategorija:**\n{least_volatile}")

        with col3:
            st.warning(f"**Analizirano obdobje:**\n{date_range.days} dni")

        # Dodatni vpogledi
        expensive_categories = volatility_df.nlargest(3, 'Povprečna cena')['Kategorija'].tolist()
        cheap_categories = volatility_df.nsmallest(3, 'Povprečna cena')['Kategorija'].tolist()

        st.write("**Najdražje kategorije:**", ", ".join(expensive_categories))
        st.write("**Najcenejše kategorije:**", ", ".join(cheap_categories))


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
