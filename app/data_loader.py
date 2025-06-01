import pandas as pd
import streamlit as st
from typing import Optional
import re


# Set za performance - O(1)
FRUITS = {
    'jabolka', 'jagode', 'pomaranče', 'banane', 'sadje', 'lubenica', 'hruške', 'limone', 'grozdje',
    'kaki', 'breskve', 'nektarine', 'borovnice', 'ananas', 'mango', 'avokado', 'kivi', 'maline',
    'češnje', 'višnje', 'melone', 'fige', 'grenivka', 'mandarine', 'kokos', 'datelj', 'lešniki',
    'orehi', 'pistacija', 'bademi', 'rozine', 'cranberry', 'ribez', 'bezeg', 'jablko', 'jabolko',
    'sliva', 'marelica', 'limeta', 'pomelo', 'pitaja', 'granatno jabolko', 'kumquat', 'satsuma',
    'citrusi', 'tropsko sadje', 'suho sadje', 'suhe marelice', 'suhe slive', 'suhe fige',
    'suhe datlje', 'suhe rozine', 'suhe brusnice', 'suhe marelice', 'suhe slive', 'suhe fige', 'borovnice',
    'goji jagode', 'acai jagode', 'kivi sadje', 'pitahaya', 'dragon fruit', 'rambutan', 'longan',
    'lychee', 'star fruit', 'jackfruit', 'tamarind', 'soursop', 'papaja', 'carambola', 'clementine', 'mandarina',
    'kumquat', 'tangerina', 'citrus', 'limonada', 'limonin sok', 'pomarančni sok', 'banana', 'brusnice'
}

VEGETABLES = {
    'solata', 'zelenjava', 'krompir', 'čebula', 'šparglji', 'paradižnik', 'zelje', 'korenje',
    'motovilec', 'brokoli', 'špinača', 'repa', 'paprika', 'bučke', 'česen', 'kumare', 'redkvice',
    'ohrovt', 'por', 'artičoke', 'beluš', 'fižol', 'grah', 'leča', 'soja', 'pesa', 'redkev',
    'rukola', 'endivija', 'cvetača', 'keleraba', 'koleraba', 'bambus', 'shiitake', 'gobe',
    'jurčki', 'lisičke', 'šampinjoni', 'rdeča pesa', 'sladki krompir', 'batata', 'riž', 'koruza',
    'ajda', 'proso', 'kvinoja', 'bulgur', 'ječmen', 'pirina kaša', 'oves', 'rižota', 'polenta',
    'špinača', 'šparglji', 'blitva', 'zeleni fižol', 'zelena', 'cikla', 'pehtran', 'bazilika',
    'origano', 'timijan', 'rožmarin', 'peteršilj', 'dill', 'koper', 'majaron', 'žajbelj',
    'melisa', 'meta', 'timijan', 'cimet', 'vanilija', 'kardamom', 'kumina', 'koriander',
    'čili', 'paprika', 'poper', 'sol', 'sladkor', 'olje', 'kis', 'limonin sok', 'jabolčni kis',
    'balzamični kis', 'olivno olje', 'sončnično olje', 'kokosovo olje', 'avokadovo olje',
    'bučno olje', 'laneno olje', 'sezamovo olje', 'orehovo olje', 'pšenično olje', 'riževo olje',
    'mandljevo olje', 'grozdno olje', 'pistacijino olje', 'makadamijino olje', 'arganovo olje',
    'črni poper', 'beli poper', 'zeleni poper', 'rožič', 'sladki krompir', 'korenje', 'buča',
    'brokoli', 'cvetača', 'špinača', 'zelena solata', 'ledena solata', 'rukola', 'endivija',
    'radicchio', 'zeleni fižol', 'šparglji', 'por', 'korenje', 'redkev', 'rukola', 'špinača',
    'motovilec', 'blitva', 'špinača', 'šparglji', 'brokoli', 'cvetača', 'korenje', 'zelena',
    'fižol', 'grah', 'leča', 'soja', 'pesa', 'redkev', 'rukola', 'endivija', 'korenje',
    'špinača', 'šparglji', 'brokoli', 'cvetača', 'korenje', 'zelena', 'fižol', 'grah',
    'leča', 'soja', 'pesa', 'redkev', 'rukola', 'endivija', 'korenje', 'špinača', 'šparglji',
    'brokoli', 'cvetača', 'korenje', 'zelena', 'fižol', 'grah', 'leča', 'soja', 'pesa',
    'redkev', 'rukola', 'endivija', 'korenje', 'špinača', 'šparglji', 'brokoli', 'cvetača',
    'korenje', 'zelena', 'fižol', 'grah', 'leča', 'soja', 'pesa', 'redkev', 'rukola',
    'endivija', 'korenje', 'špinača', 'šparglji', 'brokoli', 'cvetača', 'korenje', 'zelena',
    'fižol', 'grah', 'leča', 'soja', 'pesa', 'redkev', 'rukola', 'endivija', 'korenje',
    'špinača', 'šparglji', 'brokoli', 'cvetača', 'korenje', 'zelena', 'fižol', 'grah',
    'leča', 'soja', 'pesa', 'redkev', 'rukola', 'endivija', 'korenje', 'špinača', 'šparglji',
    'brokoli', 'cvetača', 'korenje', 'zelena', 'fižol', 'grah', 'leča', 'soja', 'pesa',
    'redkev', 'rukola', 'endivija', 'korenje', 'špinača', 'šparglji', 'brokoli', 'cvetača',
    'korenje', 'zelena', 'fižol', 'grah', 'leča', 'soja', 'pesa', 'redkev', 'rukola',
    'endivija', 'korenje', 'špinača', 'šparglji', 'brokoli', 'cvetača', 'korenje', 'zelena',
    'fižol', 'grah', 'leča', 'soja', 'pesa', 'redkev', 'rukola', 'endivija', 'korenje',
    'špinača', 'šparglji', 'brokoli', 'cvetača', 'korenje', 'zelena', 'fižol', 'grah',
    'leča', 'soja', 'pesa', 'redkev', 'rukola', 'endivija', 'korenje', 'špinača', 'šparglji'
}

MEATS_FISH = {
    'meso', 'piščanec', 'piščančji', 'govedina', 'goveje', 'hrenovke', 'svinjsko', 'file',
    'bedra', 'puranje', 'puranji', 'sardele', 'ćevapčiči', 'pleskavice', 'burger', 'lignji',
    'osliča', 'salama', 'jajca', 'tuninin', 'poli', 'račji', 'pečenice', 'vampi', 'krvavice',
    'telečji', 'tuna', 'slanina', 'klobasa', 'losos', 'postrv', 'brancin', 'orada', 'škampi',
    'rakci', 'morski sadeži', 'kaviar', 'anchovi', 'bakalar', 'sardine', 'skuša', 'ribe',
    'pršut', 'panceta', 'kebab', 'čevapi', 'kotlet', 'zrezek', 'golaž', 'mortadela', 'piščančje',
    'piščančje prsi', 'piščančji file', 'piščančje bedra', 'piščančja krila', 'piščančja rebra',
    'piščančja klobasa', 'piščančji burger', 'piščančji nuggeti', 'piščančji trakovi',
    'piščančji ražnjiči', 'piščančji kebab', 'piščančja salama', 'piščančja pašteta', 'piščančja klobasa',
    'piščančji fileti', 'piščančji zrezki', 'piščančje meso', 'piščančja prsa', 'piščančje stegno',
    'piščančje krače', 'piščančje nogice', 'piščančje perutničke', 'piščančje boke', 'piščančji vratovi',
    'piščančje ledvice', 'piščančje srce', 'piščančja jetra', 'piščančji fileti z rožmarinom',
    'piščančji fileti z limono', 'piščančji fileti z zelišči', 'piščančji fileti s česnom', 'file', 'ribe', 'riba',
    'losos', 'lososov file', 'lososova fileta', 'lososov steak', 'lososov zrezek', 'lososova plošča',
    'lososova rezina', 'lososov file z rožmarinom', 'lososov file z limono', 'lososov file z zelišči',
    'lososov file s česnom', 'lososov file z medom', 'lososov file z gorčico', 'lososov file z limono in koper',
    'lososov file z limono in peteršiljem', 'lososov file z limono in česnom', 'lososov file z limono in timijan',
    'lososov file z limono in origano', 'lososov file z limono in baziliko', 'lososov file z limono in majaron',
    'lososov file z limono in timijanom', 'lososov file z limono in rožmarinom', 'lososov file z limono in žajbljem', 'zrezki',
    'zrezek', 'goveji zrezek', 'goveji steak', 'goveji file', 'goveji burger', 'goveja rebra',
    'goveji zrezki', 'goveji fileti', 'goveja plečka', 'goveja rebra', 'goveja klobasa', 'goveja salama',
    'goveja pašteta', 'goveji kebab', 'goveji čevapi', 'goveji burger', 'goveji zrezki z rožmarinom', 'pašteta',
    'goveja pašteta', 'goveja klobasa', 'goveja salama', 'goveji kebab', 'goveji čevapi', 'goveji burger', 'burger',
    'burgerji', 'hamburger', 'hamburgerji', 'cheeseburger', 'cheeseburgerji', 'veggie burger',
    'veggie burgerji', 'piščančji burger', 'piščančji burgerji', 'goveji burger', 'goveji burgerji',
    'lososov burger', 'lososov burgerji', 'tuna burger', 'tuna burgerji', 'vegetarijanski burger',
    'vegetarijanski burgerji', 'veganski burger', 'veganski burgerji', 'čevapi', 'čevapi',
    'čevapi z rožmarinom', 'čevapi z limono', 'čevapi z zelišči', 'čevapi s česnom', 'čevapi z medom', 'odojek', 
    'goveja', 'piščančja', 'svinjska', 'puranja', 'jagnječja', 'telečja', 'raca', 'račja',
    'lososova', 'tuna', 'sardelna', 'školjke', 'lignji', 'škampi', 'rakci', 'morski sadeži',
    'morski pes', 'lososov steak', 'lososov zrezek', 'lososova plošča', 'lososova rezina', 'perutnina',
    'perutninski', 'piščančje ', 'piščančje prsi', 'piščančje stegno', 'krače', 'meso', 'fileti',
    'filet', 'fileti z rožmarinom', 'fileti z limono', 'fileti z zelišči', 'fileti s česnom'
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
    'polnozrnati', 'raženi', 'beli kruh', 'črni kruh', 'bezglutenski', 'ajdov', 'koruzni',
    'pirin', 'kukuruzni', 'prosen', 'kajzerice', 'kajzerica', 'žitni', 'švedski kruh', 'francoski kruh',
    'italijanski kruh', 'ciabatta', 'ciabatta kruh', 'baguette', 'pita kruh', 'naan',
    'lavash', 'tortilja', 'pita kruh', 'kruh brez glutena', 'kruh z oreščki', 'kruh z semeni', 'sok',
    'pomarančni sok', 'jabolčni sok', 'grozdni sok', 'ananasov sok', 'limonin sok', 'limonada',
    'jagodni sok', 'borovničevo sok', 'malinov sok', 'breskov sok', 'marelični sok',
    'granatno jabolko sok', 'kivi sok', 'mango sok', 'tropski sok', 'zeleni sok', 'smoothie',
    'smoothie bowl', 'smoothie napitek', 'smoothie zeleni', 'smoothie sadni', 'smoothie jagodni',
    'smoothie malinov', 'smoothie borovničevo', 'smoothie ananasov', 'smoothie breskov',
    'smoothie marelični', 'smoothie granatno jabolko', 'smoothie kivi', 'smoothie mango', 'smoothie', 'cedevita',
    'gazirana voda', 'mineralna voda', 'brezalkoholno pivo', 'pivo brez alkohola', 'energijska pijača',
    'energijski napitek', 'kava', 'čaj', 'čaj z limono', 'čaj z medom', 'čaj z ingverjem',
    'čaj z meto', 'čaj z limono in medom', 'čaj z limono in ingverjem', 'čaj z limono in meto',
    'coca cola', 'coca cola zero', 'coca cola light', 'fanta', 'sprite',
    'pepsi', 'pepsi max', 'pepsi light', 'red bull', 'red bull zero', 'red bull sugar free',
    'sprite zero', 'sprite light', 'fanta zero', 'fanta light', 'limonada', 'limonada brez sladkorja',
    'limonada z ingverjem', 'limonada z meto', 'limonada z limono', 'limonada z jagodami',
    'coca', 'cola', 'coca cola', 'coca cola zero', 'coca cola light', 'fanta', 'sprite'
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
        df = pd.read_csv('../vse_cene_grouped_top6_fixed.csv', sep=';', encoding='utf-8')

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
        df['Is_Slovenian'] = df['Product Name'].str.contains('slovensko|slovenski|slovenska', case=False, na=False)

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