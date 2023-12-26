import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import SQLAlchemyError
from transliterate import slugify
from transliterate import detect_language
from thefuzz import process
from dotenv import load_dotenv

engine = None

assert load_dotenv()

# Получаем данные из файла .env
USR = os.getenv("USR")
PWD = os.getenv("PWD")
DB_HOST = os.getenv("DB_HOST")
PORT = os.getenv("PORT")
DB = os.getenv("DB")

DATABASE = {
    "drivername": "postgresql",
    "username": USR,
    "password": PWD,
    "host": DB_HOST,
    "port": PORT,
    "database": DB,
    "query": {},
}

# Создание объекта Engine
engine = create_engine(URL.create(**DATABASE))

# Проверка соединения
try:
    with engine.connect() as conn:
        # Попытка выполнить простой тестовый запрос. Функция `text` преобразует строку в SQL-запрос
        result = conn.execute(text("SELECT 1"))
        for _ in result:
            pass
    print(f"Connection established: {DATABASE['database']} на {DATABASE['host']}")
except SQLAlchemyError as e:
    print(f"Connection error: {e}")


def load_the_data(
    country_selection=["RU", "KZ", "AM", "RS", "ME", "KG", "GE"], expanded=False
):
    if expanded:
        table = "geonames"
    else:
        table = "cities15000"
    where_clause = ""
    if country_selection:
        if len(country_selection) == 1:
            where_clause = f"WHERE country_code = '{country_selection[0]}'"
        else:
            where_clause = f"WHERE country_code IN {tuple(country_selection)}"

    query = f"""
    SELECT geonameid, name, alternatenames, country_code
    FROM {table}  
    {where_clause}
    """

    # Загружаем данные
    global engine
    df = pd.read_sql_query(query, con=engine, index_col="geonameid")

    altnames = [l.split(",") if l else [None] for l in df.alternatenames.values]
    names = df.name.values

    for i in range(len(altnames)):
        altnames[i].append(names[i])
     d = {ind: n for ind, n in zip(df.index, altnames)}
    return d

def search(
    query: str,
    k=10,
    weight_mode="exp",
    asdict=True,
    country_selection=["RU", "KZ", "AM", "RS", "ME", "KG", "GE"],
):

    global engine

    d = load_the_data(country_selection)

    if detect_language(query) is not None:
        query = slugify(query)
    scores = {}

    for (
        ind,
        name_list,
    ) in (
        d.items()
    ):  # для каждого города вычислим оценки схожести со всеми альтернативными названиями
    _ = np.array(process.extract(query, name_list))
        _ = np.array(process.extract(query, name_list))

        if weight_mode == "exp":
            # Вычислим оценки и взвесить их экспоненциально
            scores[ind] = np.exp(_[:, 1].astype(int)).sum() / len(
                _
            )   # суммируемь экспоненты оценок и нормализовать по числу возможных названий
            scores[ind] = np.log(scores[ind])  # return to the readable values
        elif weight_mode == "sq":
            # Вычислим оценки и взвесить их параболически
            scores[ind] = np.square(_[:, 1].astype(int)).sum() / len(_)
            scores[ind] = np.sqrt(scores[ind])
        else:
            scores[ind] = _[:, 1].astype(int).sum() / len(_)


# отсортировано по score
    scores_df = pd.DataFrame.from_records(
        sorted(scores.items(), key=lambda item: item[1], reverse=True),
        columns=["geonameid", "score"],
    )
    scores_df.loc[:, "score"] = scores_df.loc[:, "score"].round(
        3
    )

    indexes = tuple(
        scores_df.loc[: k - 1, "geonameid"]
    )  # выбираем индексы DataFrame для первых k значений

    query = f"""
        SELECT DISTINCT
            cities.geonameid,
            cities.name,
            regions.name as region,
            ci."Country" as country

        FROM
            cities15000 AS cities
        LEFT JOIN
            (SELECT "ISO", "Country" FROM "countryInfo") AS ci
        ON
            cities.country_code = ci."ISO"
        LEFT JOIN
            "admin1CodesASCII" AS regions
        ON
            COALESCE(cities.country_code, '') || '.' || COALESCE(cities.admin1_code, '') = regions.code
        WHERE
            cities.geonameid IN {indexes};
    """

    qres = pd.read_sql_query(query, con=engine)
    result = (
        pd.merge(qres, scores_df, on="geonameid", how="left")
        .sort_values("score", ascending=False)
        .set_index("geonameid")
    )
    if asdict:
        return result.T.to_dict()
    else:
        return result
