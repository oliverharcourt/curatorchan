import logging
import time

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def make_request(url, query, variables, max_failures=5, max_backoff=60):
    failures = 0
    backoff = 3

    while failures < max_failures:
        try:
            response = requests.post(url, json={"query": query, "variables": variables})
            if response.status_code == 200:
                return response.json()
            logger.error(
                f"Request failed with status code {response.status_code}: {response.text}"
            )
        except Exception as e:
            logger.error(f"Error during post request: {e}")

        failures += 1
        time.sleep(backoff)
        backoff = min(backoff * 2, max_backoff)

    raise RuntimeError("Max failures reached while making request")


def run_query(query, query_type, init_page):
    API_URL = "https://graphql.anilist.co"

    variables = {"page": init_page, "perPage": 100}

    pbar = tqdm(desc="Fetching data", unit="pages")

    response = requests.post(API_URL, json={"query": query, "variables": variables})
    if response.status_code != 200:
        raise Exception(
            f"Initial query failed with status code {response.status_code}: {response.text}"
        )
    data = response.json()["data"]["Page"][query_type]
    has_next_page = response.json()["data"]["Page"]["pageInfo"]["hasNextPage"]
    pbar.update(1)

    while has_next_page:
        variables["page"] += 1
        try:
            response = make_request(
                API_URL, query, variables, max_failures=5, max_backoff=60
            )
        except RuntimeError as e:
            logger.error(f"Error fetching data: {e}")
            save_data(data, f"{query_type}.csv")
            variables["page"] -= 1
            has_next_page = True  # Continue to retry the same page
            continue

        new_data = response["data"]["Page"][query_type]
        has_next_page = response["data"]["Page"]["pageInfo"]["hasNextPage"]
        data.extend(new_data)
        pbar.update(1)

        if variables["page"] % 1000 == 0:
            logger.info(f"Fetched {len(data)} {query_type} records so far.")
            save_data(data, f"{query_type}.csv")
            logger.info(f"Saved {query_type} data to {query_type}.csv")
        time.sleep(3)

    pbar.close()

    save_data(data, f"{query_type}.csv")


def save_data(data, filename):
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(filename, index=False, mode="w")


if __name__ == "__main__":
    user_query = """
    query UserQuery($page: Int, $perPage: Int) {
    Page(page: $page, perPage: $perPage) {
        pageInfo {
        hasNextPage
        }
        users {
        id
        name
        updatedAt
        statistics {
            anime {
            meanScore
            minutesWatched
            scores {
                score
                count
                meanScore
                mediaIds
            }
            statuses {
                status
                count
                mediaIds
            }
            genres {
                genre
                count
                meanScore
                minutesWatched
            }
            }
        }
        }
    }
    }
    """

    anime_query = """
    query AnimeQuery($page: Int, $perPage: Int) {
    Page(page: $page, perPage: $perPage) {
        pageInfo {
        hasNextPage
        }
        media {
        id
        title {
            native
            romaji
            english
        }
        description
        genres
        tags {
            id
            name
            category
            isAdult
            description
            rank
        }
        isAdult
        format
        meanScore
        popularity
        relations {
            nodes {
            id
            title {
                english
            }
            }
        }
        startDate {
            year
        }
        endDate {
            year
        }
        season
        updatedAt
        stats {
            scoreDistribution {
            amount
            score
            }
        }
        }
        }
    }
    }
    """

    run_query(user_query, query_type="users", init_page=1)
    run_query(anime_query, query_type="media", init_page=1)

    print("Data fetched and saved!")
