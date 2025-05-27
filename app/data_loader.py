import pandas as pd
import streamlit as st
from typing import Optional
import re


# Set za performance - O(1)
FRUITS = {
    'jabolka', 'jagode', 'pomaranče', 'banane', 'sadje', 'lubenica', 'hruške', 'limone', 'grozdje',
    'kaki', 'breskve', 'nektarine', 'borovnice', 'ananas', 'mango', 'avokado', 'kivi', 'maline',
    'češnje', 'višnje', 'melone', 'fige', 'grenivka', 'mandarine', 'kokos', 'datelj', 'lešniki',
    'orehi', 'pistacija', 'bademi', 'rozine', 'cranberry', 'ribez', 'bezeg', 'jablko'
}

VEGETABLES = {
    'solata', 'zelenjava', 'krompir', 'čebula', 'šparglji', 'paradižnik', 'zelje', 'korenje',
    'motovilec', 'brokoli', 'špinača', 'repa', 'paprika', 'bučke', 'česen', 'kumare', 'redkvice',
    'ohrovt', 'por', 'artičoke', 'beluš', 'fižol', 'grah', 'leča', 'soja', 'pesa', 'redkev',
    'rukola', 'endivija', 'cvetača', 'keleraba', 'koleraba', 'bambus', 'shiitake', 'gobe',
    'jurčki', 'lisičke', 'šampinjoni', 'rdeča pesa', 'sladki krompir', 'batata'
}

MEATS_FISH = {
    'meso', 'piščanec', 'piščančji', 'govedina', 'goveje', 'hrenovke', 'svinjsko', 'file',
    'bedra', 'puranje', 'puranji', 'sardele', 'ćevapčiči', 'pleskavice', 'burger', 'lignji',
    'osliča', 'salama', 'jajca', 'tuninin', 'poli', 'račji', 'pečenice', 'vampi', 'krvavice',
    'telečji', 'tuna', 'slanina', 'klobasa', 'losos', 'postrv', 'brancin', 'orada', 'škampi',
    'rakci', 'morski sadeži', 'kaviar', 'anchovi', 'bakalar', 'sardine', 'skuša', 'ribe',
    'pršut', 'panceta', 'kebab', 'čevapi', 'kotlet', 'zrezek', 'golaž', 'mortadela'
}

DAIRY = {
    'mleko', 'sir', 'jogurt', 'maslo', 'skuta', 'actimel', 'kefir', 'smetana', 'kisla smetana',
    'parmezan', 'mocarela', 'gorgonzola', 'camembert', 'brie', 'cheddar', 'feta', 'cottage',
    'ricotta', 'skyr', 'grški jogurt', 'probiotik', 'lakto', 'brez laktoze', 'sojino mleko',
    'mandljevo mleko', 'ovseno mleko', 'kokosovo mleko', 'rženo mleko', 'ajdovo mleko'
}

DRINKS_BREAD = {
    'kruh', 'pivo', 'vino', 'cedevita', 'pepsi', 'pijača', 'napitek', 'štručka', 'smoothie',
    'kombucha', 'žemlja', 'sok', 'voda', 'coca cola', 'fanta', 'sprite', 'red bull', 'kava',
    'čaj', 'limonada', 'mineralna', 'gazirana', 'brez sladkorja', 'energijska', 'poper',
    'toast', 'bagel', 'croissant', 'brioche', 'focaccia', 'pita', 'tortilla', 'wrap',
    'polnozrnati', 'raženi', 'beli kruh', 'črni kruh', 'bezglutenski', 'ajdov'
}

SWEETS_SNACKS = {
    'čokolada', 'piškoti', 'torta', 'sladkarije', 'bonboni', 'gumi medvedki', 'lizike',
    'čips', 'popcorn', 'oreščki', 'slani', 'krekri', 'napolitanke', 'vaflje', 'muffin',
    'brownies', 'cookie', 'donut', 'praline', 'nugat', 'karamel', 'žvečilni', 'tic tac',
    'haribo', 'kit kat', 'snickers', 'mars', 'twix', 'bounty', 'milka', 'lindt'
}

HYGIENE_CLEANING = {
    'čistilo', 'prašek', 'šampon', 'milo', 'pasta za zobe', 'deodorant', 'parfum', 'krema',
    'toaletni papir', 'brisače', 'obvezice', 'kondomi', 'detergent', 'mehčalec', 'belilo',
    'dezinfekcija', 'disinfectant', 'antibakterijski', 'wc', 'kopalnica', 'kuhinja',
    'pomivalno sredstvo', 'splakovalec', 'osvežilec', 'dišava', 'čistilni', 'higiena'
}

SPICES_CONDIMENTS = {
    'začimbe', 'sol', 'poper', 'olje', 'kis', 'sladkor', 'med', 'marmelada', 'džem',
    'kečap', 'majoneza', 'gorčica', 'tartar', 'pesto', 'ajvar', 'harisa', 'wasabi',
    'tabasco', 'chili', 'paprika prah', 'cimet', 'vanilija', 'bazil', 'origano',
    'timijan', 'rožmarin', 'peteršilj', 'dill', 'kumin', 'kari', 'ingver', 'kurkuma'
}

FROZEN_FOODS = {
    'zamrznjeno', 'frozen', 'sladoled', 'gelato', 'pizza', 'lasagna', 'špinača zamrznjena',
    'grah zamrznjen', 'jagode zamrznjene', 'smoothie pak', 'ice cream', 'sorbet'
}

BABY_PRODUCTS = {
    'otroška', 'baby', 'dojenček', 'plenice', 'mleko za dojenčke', 'kašica', 'otroški',
    'hipp', 'nestle', 'aptamil', 'bebivita', 'frutek'
}

# Pakiranje in merjenje za boljše ujemanje
PACKAGING_WORDS = {
    'pakirano', 'pakiran', 'fresh', 'bio', 'eko', 'organic', 'naravno', 'domače',
    'slovensko', 'slovenski', 'prva', 'izbor', 'premium', 'extra', 'special'
}

# Teža/velikost vzorec
MEASUREMENT_PATTERN = re.compile(r'\b\d+\s*(g|kg|ml|l|cl|dl)\b', re.IGNORECASE)
PRICE_PATTERN = re.compile(r'\b\d+[,.]?\d*\s*€?\b')


@st.cache_data
def load_and_process_data() -> Optional[pd.DataFrame]:
    """Procesiraj in naloži datoteko"""
    try:
        df = pd.read_csv('../vsikatalogi_ready_no_outliers_topstores_grouped.csv', sep=';', encoding='utf-8')

        # Osnovno pred procesiranje
        df['Date'] = pd.to_datetime(df['Date'])
        df['Price'] = df['Price (€)']

        # Odstrani neuspešne pretvorbe
        df = df.dropna(subset=['Price'])

        df['Month'] = df['Date'].dt.to_period('M')
        df['Week'] = df['Date'].dt.to_period('W')
        df['DayOfWeek'] = df['Date'].dt.day_name()

        # Product kategorija
        df['Category'] = df['Product Name'].apply(categorize_product)

        # Dodatni stolpci za analizo
        df['Has_Discount'] = df['Product Name'].str.contains('1\+1|2\+1|50%|akcija', case=False, na=False)
        df['Is_Bio'] = df['Product Name'].str.contains('bio|eko|organic', case=False, na=False)
        df['Is_Slovenian'] = df['Product Name'].str.contains('slovensko|slovenski', case=False, na=False)

        # Statistika za grupiranja
        df['Is_Grouped'] = df['Product Name'] != df['Product Name Grouped']

        return df

    except Exception as e:
        st.error(f'Error {e}')
        return None


def clean_product_name(product_name: str) -> str:
    """Čisto ime za boljšo kategorizacijo"""
    if not isinstance(product_name, str):
        return ""

    cleaned = product_name.lower().strip()

    # Odstrani besede za pakiranje
    for word in PACKAGING_WORDS:
        cleaned = cleaned.replace(word, ' ')

    # Odstrani meritve in cene
    cleaned = MEASUREMENT_PATTERN.sub('', cleaned)
    cleaned = PRICE_PATTERN.sub('', cleaned)

    # Počisti presledke
    cleaned = ' '.join(cleaned.split())

    return cleaned


def categorize_product(product_name: str) -> str:
    """Kategoriziraj vse produkte za lažji pregled"""
    if pd.isna(product_name) or not isinstance(product_name, str) or len(product_name) == 0:
        return 'Ostalo'

    # Originalno in nova verzija
    original = product_name.lower().strip()
    cleaned = clean_product_name(product_name)

    # Razdeli v besede za boljše ujemanje
    original_words = set(original.split())
    cleaned_words = set(cleaned.split())
    all_words = original_words | cleaned_words

    if all_words & FRUITS:
        return 'Sadje'
    elif all_words & VEGETABLES:
        return 'Zelenjava'
    elif all_words & MEATS_FISH:
        return 'Meso in ribe'
    elif all_words & DAIRY:
        return 'Mlečni izdelki'
    elif all_words & DRINKS_BREAD:
        return 'Pijače in kruh'
    elif all_words & SWEETS_SNACKS:
        return 'Sladkarije in prigrizki'
    elif all_words & HYGIENE_CLEANING:
        return 'Higiena in čiščenje'
    elif all_words & SPICES_CONDIMENTS:
        return 'Začimbe in dodatki'
    elif all_words & FROZEN_FOODS:
        return 'Zamrznjena hrana'
    elif all_words & BABY_PRODUCTS:
        return 'Otroški izdelki'

    # Dodatna kategorizacija
    full_text = f"{original} {cleaned}"

    # Gospodinjski izdelki
    if any(pattern in full_text for pattern in ['kuhinja', 'posoda', 'kozarec', 'krožnik', 'pribor']):
        return 'Gospodinjski izdelki'

    # Zdravje in farmacija
    elif any(pattern in full_text for pattern in ['vitamin', 'mineral', 'zdravil', 'lekarna', 'prehranski dodatek']):
        return 'Zdravje in lekarništvo'

    # Izdelki za živali
    elif any(pattern in full_text for pattern in ['hrana za', 'pasja', 'mačja', 'pes', 'mačka']):
        return 'Hrana za živali'

    return 'Ostalo'


def get_data_info(df: pd.DataFrame) -> dict:
    """Vrni informacije o grupiranju podatkov"""
    total_products = len(df['Product Name'].unique())
    grouped_products = len(df['Product Name Grouped'].unique())

    return {
        'total_rows': len(df),
        'unique_products': total_products,
        'grouped_products': grouped_products,
        'grouping_effect': total_products - grouped_products,
        'stores': sorted(df['Store'].unique().tolist())
    }