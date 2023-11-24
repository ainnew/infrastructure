import requests

import numpy as np
import pandas as pd
import geopandas as gpd

import pdfplumber

from scipy import stats


from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import missingno as msno

from fuzzywuzzy import fuzz


# Функция получения данных по api с data.mos.ru
def get_api_data_mos_ru(api_key, domain, dataset_id):
    """
    Функция обращается по api для получения данных с помощью GET-запросов.
    Если количество данных превышает 1000, то запрос отправляется в цикле, каждый раз скачивая порцию данных с шагом в 1000 записей
    Работа функции сопровождается выводом текстовой информации об объеме данных для скачивания и прогрессе
    Параметры:
        api_key : ключ пользователя
        domain : домен ресурса
        dataset_id : идентификатор датасета для скачивания
    Выводит:
        текстовая информация
    Возвращает:
        list : список с полученными данными
    """

    print(f'Датасет №{dataset_id}')
    url_count = f'{domain}{dataset_id}/count'
    url_data = f'{domain}{dataset_id}/rows' # rows features

    if dataset_id == '60622':
        params = {'api_key' : api_key, '$filter' : "Cells/Address eq '2'"} # для спортивных залов сразу выбираем данные с условием, как вариант
    else:
        params = {'api_key' : api_key}

    result_count = requests.get(url_count, params = params)
    if result_count.status_code == 200:
        result_count_json = result_count.json()
        print('Изначально элементов в датасете:', result_count_json)
        if result_count_json > 1000:
            result_data_json = []
            for i in range(int(result_count_json / 1000) + 1):
                params['$top'] = 1000
                params['$skip'] = i * 1000
                result_data = requests.get(url_data, params = params)
                result_data_json.extend(result_data.json())
                print(len(result_data.json()))
        else:
            result_data = requests.get(url_data, params = params)
            result_data_json = result_data.json()
            print(len(result_data.json()))

        print('Получено элементов:', len(result_data_json), end = '\n\n')
        return result_data_json
    else:
        print('Статус ответа:', result_count.status_code)



# Функция поиска необходимых атрибутов данных data.mos.ru
def _extract_attrs(dataset, dataset_id):
    """
    Вспомогательная функция для функции get_df
    """
    attrs = []
    for data in dataset:
        attrs.append({
            'district': data['Cells']['AdmArea'],
            'region': data['Cells']['District'],
            'lonlat': data['Cells']['geoData']['coordinates'],
            'capacity': data['Cells']['Capacity'] if dataset_id == '916' else np.NaN,
            'light': data['Cells']['Lighting'] if dataset_id == '2663' else np.NaN,
            'addr': data['Cells']['Address'] if dataset_id == '60622' else np.NaN,
        })
    return attrs

# Функции формирования датафрейма на основе атрибутов и создания в нем столбца геометрии из координат
def get_df(dataset, dataset_id):
    """
    Функция формирует датафрейм на основе атрибутов из dataset
    Параметры:
        dataset : набор данных
        dataset_id : идентификатор набора данных
    Возвращает:
        df : датафрейм
    """
    attrs = _extract_attrs(dataset, dataset_id)
    df = pd.DataFrame(attrs)
    df = df.dropna(axis=1)
    return pd.DataFrame(df)


def set_geo(df):
    """
    Функция создает столбцы 'lon' и 'lat' из столбца 'lonlat' входного датафрейма df, затем создает геометрический столбец с помощью функции points_from_xy.
    Параметры:
        df : датафрейм
    Возвращает:
        df : датафрейм с геометрическим столбцом
    """
    df[['lon', 'lat']] = pd.DataFrame(df['lonlat'].tolist(), index=df.index)
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs='epsg:4326') # создаем геометрический столбец с помощью функции points_from_xy
    df = df.drop(columns=['lonlat', 'lon', 'lat'])
    return df


# Функция чтения pdf-файла и объединения данных таблицы со всех страниц присутствия в один датафрейм
def get_pdf_table(path, pages_slice):
    """
    Функция читает pdf-файл по указанному пути и объединяет данные таблицы со всех страниц в один датафрейм.
    Параметры:
        path : путь к файлу
        pages_slice : срез страниц
    Возвращает:
        df : датафрейм
    """
    with pdfplumber.open(path) as pdf:
        table_data = []
        for page in pdf.pages[pages_slice[0]:pages_slice[1]]:
            table = page.extract_table()
            table_data.extend(table)
    return pd.DataFrame(table_data[1:], columns=table_data[0])

# Функция чтения GeoJSON и создания GeoDataFrame на его основе
def get_geo_from_json(path, col):
    """
    Функция читает геоданные из json-файла по указанному пути и объединяет их в геодатафрейм.
    Параметры:
        path : путь к файлу
        col : столбец
    Возвращает:
        df : геодатафрейм
    """
    df_geo = gpd.GeoDataFrame()
    for f in [path]:
        gdf = gpd.read_file(f, crs='epsg:4326')
        # gdf = gdf[gdf[col]=='smth']
        gdf = gdf[[col,'geometry']]
        df_geo = pd.concat([df_geo, gdf])
    return df_geo




# Функции описания данных
def _show_isna_small(df):
    """
    Вспомогательная функция вывода графиков для функции describe_isna
    """
    fig = plt.figure(figsize=(18,4))
    ax1 = fig.add_subplot(1,2,1)
    msno.bar(df, color='Blue', fontsize=8, ax=ax1) 
    ax2 = fig.add_subplot(1,2,2)

    # print(df.select_dtypes(include=['number']).columns)
    msno.heatmap(df.select_dtypes(include=['number']), cmap=LinearSegmentedColormap.from_list("", ['Blue', 'white', "DeepPink"]), fontsize=8, ax=ax2)
    plt.tight_layout()
    
    fig = plt.figure(figsize=(18,4))
    ax3 = fig.add_subplot(1,2,1)
    sns.heatmap(df.isnull(), cmap=sns.color_palette(['Blue', 'white']), ax=ax3)
    ax4 = fig.add_subplot(1,2,2)
    msno.dendrogram(df, fontsize=8, ax=ax4)
    plt.tight_layout()


def describe_isna(df, show = True):
    """
    Функция выводит форму, размер датафрейма и общую информацию о пропущенных значениях, а также данные о количестве и долях пропущенных значений в каждом столбце.
    Вывод сопровождается четырьмя графиками (barplot, heatmap , matrix plot, dendrogram из библиотеки missingno), два последних из которых выводятся опционально.
    Параметры:
        df : датафрейм
        show_matrix : bool (default True), опциональный вывод графиков matrix plot, dendrogram
    Выводит:
        текстовая информация
    Возвращает:    
        графики barplot, heatmap , matrix plot, dendrogram
        df : датафрейм
    """
    size = df.size
    isna = df.isna().sum().sum()
    print('shape', df.shape)
    print('size', size)
    print('isna', isna)
    print('isna share {:.2%}'.format(isna / size))
    df_isna = pd.concat([df.dtypes, df.count(), df.isna().sum(), (df.isna().sum() / df.shape[0]).map('{:.2%}'.format)], axis=1).rename(columns = {0 : 'dtype', 1 : 'size', 2 : 'isna', 3 : 'isna_share'})
    display(df_isna)
    if show:
        _show_isna_small(df)


# Функции для предварительного просмотра общей статистики по данным
def describe_data(df):
    """
    Функция выводит общую статистическую информацию по нумерическим и ненумерическим данным
    Нумерические данные: count, mean, std, min, max, перцентили (25%, 50%, 75%)
    Ненумерические данные: count, unique, top, freq (частота самого популярного значения)
    Параметры:
        df : датафрейм
    Выводит:
        df : датафрейм
    """
    n_cols = df.select_dtypes(include=['number']).columns
    if not n_cols.empty:
        display(df[n_cols].describe().apply(lambda x: x.map('{:.2f}'.format)))
    o_cols = df.select_dtypes(include=['O']).columns
    if not o_cols.empty:
        display(df[o_cols].describe())


# Функция анализа распределения данных
def analyze(col):
    """
    Функция демонстрации распределения данных
    Параметры:
        col : столбец датафрейма
    Показывает:
        график барплот значений
        показатели средней, дисперсии, асимметрии, эксцесса,  теста на нормальность данных Шапиро-Уилка
    """
    # plt.style.use('ggplot')
    plt.hist(col, bins=60, color = 'Blue', alpha = .8)
    plt.title(f"{col.name.title()}\n\nmedian : {round(np.median(col), 2)}, mean : {round(np.mean(col), 2)}, std : {round(np.std(col), 2)}, skew : {round(stats.skew(col), 2)}, kurt : {round(stats.kurtosis(col), 2)}, shapiro : {round(stats.shapiro(col)[0], 2)}\n", fontdict = {'fontsize' : 8})
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)



# Функция матчинга точек объектов с полигонами регионов по геоданным
def geomatching(df1, df2):
    """
    Функция объединяет данные методом пространственного джойна на основе геометрий точек и полигонов
    Параметры:
        df1 : первый датафрейм с геометрическим столбцом
        df2 : второй датафрейм с геометрическим столбцом
    Возвращает:
        df_join : датафрейм с данными из двух датафреймов, где каждая строка является описанием точки
    """
    df_join = gpd.sjoin(df1, df2, how="inner", predicate='contains')
    return df_join


# Функция для сравнения частичного совпадения текстов по расстоянию Левенштейна с помощью метода partial_ratio 
def _compare_texts(text1, text2):
    """
    Вспомогательная функция для функции find_region_diff.
    Функция сравнивает два текста и возвращает коэффициент сходства (чем ближе к 100, тем тексты более похожи)
    """
    return fuzz.partial_ratio(text1, text2)

# Функция сравнения корректности попадания точки в район по коордмнатам
def find_region_diff(df1, df2):
    """
    Функция принимает два датафрейма df1 и df2, затем объединяет их с помощью функции geomatching. Далее она вычисляет коэффициент сходства для каждой строки и выводит несоответствия и их количество.
    Параметры:
        df1 : первый датафрейм с геометрическим столбцом
        df2 : второй датафрейм с геометрическим столбцом
    Выводит:
        text : несоответствия и их количество
    """
    df_join = geomatching(df1, df2)
    df_join['similarity'] = df_join.apply(lambda row: _compare_texts(row['region_left'], row['region_right']), axis=1)
    display(df_join[df_join['similarity'] < 100].groupby(['region_left', 'region_right']).size()) # Выводим несоответствия и их количество