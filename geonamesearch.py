import pandas as pd
import numpy as np

from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import SQLAlchemyError

from transliterate import slugify
from transliterate import detect_language

from thefuzz import process
import numpy as np

from dotenv import load_dotenv
import os

engine = None

assert load_dotenv()  # Raises AssertionError if .env is not there

# Take the data from the .env file
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

# Creating an Engine object
engine = create_engine(URL.create(**DATABASE))

# Checking the connetion
try:
    with engine.connect() as conn:
        # Trying to execute a simple test query. The `text` function converst a string into and SQL-query
        result = conn.execute(text("SELECT 1"))
        for _ in result:
            pass  # don't do anything
    print(f"Connection established: {DATABASE['database']} на {DATABASE['host']}")
except SQLAlchemyError as e:
    print(f"Connection error: {e}")


def load_the_data(
    country_selection=["RU", "KZ", "AM", "RS", "ME", "KG", "GE"], expanded=False
):
    """Loads the vectors of geonameIDs, names and alternatenaes from the database. Reformats the data into the dictionary compatible with `thefuzz`"""
    if expanded:
        table = "geonames"
    else:
        table = "cities15000"
    where_clause = ""
    if country_selection:
        # country_selection = list(country_selection)
        if len(country_selection) == 1:
            where_clause = f"WHERE country_code = '{country_selection[0]}'"
        else:
            where_clause = f"WHERE country_code IN {tuple(country_selection)}"

    query = f"""
    SELECT geonameid, name, alternatenames, country_code
    FROM {table}  
    {where_clause}
    """
    # print(query)
    # print(country_selection, len(country_selection))

    # Load the data into a pandas DataFrame
    global engine
    df = pd.read_sql_query(query, con=engine, index_col="geonameid")

    altnames = [l.split(",") if l else [None] for l in df.alternatenames.values]
    names = df.name.values

    for i in range(len(altnames)):
        altnames[i].append(names[i])
    # Creating the dictionary of the structure geonameID: names for all cities including official and alternative names
    # d = {
    #     zip(df.index, altnames)
    # }  # does same as
    d = {ind: n for ind, n in zip(df.index, altnames)}
    return d


def search(
    query: str,
    k=10,
    weight_mode="exp",
    asdict=True,
    country_selection=["RU", "KZ", "AM", "RS", "ME", "KG", "GE"],
):
    """The rapid fuzzy search for city names. The matches are defined based
    on the Levenshtein distance. The function takes a query string and returns
    k best matches. Supports parabolic and exponential weighting.

    Parameters:
    - query (str): the query string
    - k (int, optional): Desired number of matches
    - weight_mode={None, 'sq', 'exp'} (str, optional):
        * None: do not weight closer matches
        * 'sq': apply parabolic weighting to the similarity scores
        * 'exp': apply exponential weighting to the similarity scores
    - asdict (bool, optional): whether to convert the result into a python dict. If False, returns Pandas DataFrame
    - country_selection (tuple): Customize the area of search. Takes ISO country codes. Does NOT handle exceptions!
    Set to None to searh around the globe.

    Returns:
    - pd.DataFrame | dict: A DataFrame or dictionary the top matching cities,
      including geonameid as index, name, country name and the region.
      Sorted by the matching score in descending order.
    """
    global engine

    d = load_the_data(country_selection)

    if detect_language(query) is not None:
        query = slugify(query)
    scores = {}  # container for match scores for each city

    for (
        ind,
        name_list,
    ) in (
        d.items()
    ):  ## for each city calculate similarity scores with every alternative name
        _ = np.array(process.extract(query, name_list))

        if weight_mode == "exp":
            # Calculate the scores and weighting them exponentially
            scores[ind] = np.exp(_[:, 1].astype(int)).sum() / len(
                _
            )  # sum up the exponents of the scores and normalize by the number of possible names
            scores[ind] = np.log(scores[ind])  # return to the readable values
        elif weight_mode == "sq":
            # Calculate the scores and weighting them parabolically
            scores[ind] = np.square(_[:, 1].astype(int)).sum() / len(_)
            scores[ind] = np.sqrt(scores[ind])
        else:  # @TODO: check for the actually None value?
            scores[ind] = _[:, 1].astype(int).sum() / len(_)

    # sorted by the matching score (.2 ms faster with the native Python function than with Pandas df.sort_values) see tests below
    scores_df = pd.DataFrame.from_records(
        sorted(scores.items(), key=lambda item: item[1], reverse=True),
        columns=["geonameid", "score"],
    )
    scores_df.loc[:, "score"] = scores_df.loc[:, "score"].round(
        3
    )  # so this looks nicer

    indexes = tuple(
        scores_df.loc[: k - 1, "geonameid"]
    )  # select the DataFrame indicies of the top k
    # print(indexes)

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
