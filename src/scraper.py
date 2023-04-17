""" Retrieves the reviews from Robert Parker and write them to file. """

import cred
import re
import os
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from pydantic import BaseSettings
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from settings import settings, logger


def open_webdriver() -> webdriver:
    """Open Chrome webdriver to scrape reviews"""
    try:
        driver = webdriver.Chrome()
    except:
        logger.error(f"Error loading the Chromedriver. Make sure to have Make sure to have Chromium / Google Chrome installed")
        raise ImportError
    return driver


def log_in_website(driver: webdriver, base_url: str) -> webdriver:
    """Login to Robert Parker's website using credentials from 'cred' file"""
    driver = open_webdriver()
    sign_in_url = base_url + "/sign-in"
    try:
        driver.get(sign_in_url)
    except:
        logger.error(f"Login page could not be loaded. Please try again.")
        raise TypeError

    # Wait max 10 seconds until at-field-username_and_email is found in the HTML code (field used to login)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "at-field-username_and_email"))
        )
    except:
        logger.error(f"Login page could not be loaded. Please try again.")
        raise TypeError

    if not "cred.py" in os.listdir():
        logger.error(f"cred.py is not availble in current directory.")
        raise ImportError

    username = cred.username
    password = cred.password

    driver.find_element("id", "at-field-username_and_email").send_keys(username)
    driver.find_element("id", "at-field-password").send_keys(password)

    driver.find_element(
        By.CSS_SELECTOR, ".btn-primary[data-loading-text='Processing…']"
    ).click()

    # Wait max 20 seconds until hello-role is found in the HTML code (only visible after successful login)
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".hello-role"))
        )
        logger.info("All done logging in. Ready to get the reviews")
    except:
        logger.error(f"Homepage could not be loaded. Please try again.")
        raise TypeError
    return driver


def construct_reviews_url(base_url: str, country) -> str:
    """Construct reviews URL with given country and base URL"""
    url = rf"{base_url}/search/wines?expand=true&show-tasting-note=true&country[]={country}"
    return url


def construct_page_urls(driver: webdriver, base_url: str, country: str) -> list[str]:
    """Construct all page URLs for the search results"""

    url = construct_reviews_url(base_url, country)
    driver.get(url)
    # Wait max 20 seconds until the reviews are loaded on the website
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".control-label"))
        )
    except:
        logger.info(f"No results found. Make sure there are results for the country requested.")

    page_html = get_html(driver)
    max_page_html = page_html.find("a", href=True, attrs={"aria-label": "Last"})["href"]
    regex = re.compile("(\d+)$")
    num_of_pages = int(re.search(regex, max_page_html).group())

    page_urls = [f"{url}&page={i}" for i in range(1, num_of_pages + 1)]
    return page_urls


def get_html(driver: webdriver) -> BeautifulSoup:
    """Request the HTML from the driver and parse it as BeautifulSoup"""
    return BeautifulSoup(driver.page_source, "html.parser")


def search_reviews(driver: webdriver, page_urls: list[str]) -> list[str]:
    """Return a list with every review in HTML"""
    reviews_html = []

    for i, url in enumerate(page_urls, start=1):
        logger.info(f"Requesting page {i} of {len(page_urls)}...") if i % 25 == 0 else None
        driver.get(url)
        # Wait max 20 seconds until the reviews are loadded on the website
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".control-label"))
            )
        except:
            logger.info(f"No results found on page {i} - Continue without ratings.")
            break
        html = get_html(driver)
        reviews_html.extend(html.find_all("div", class_="tasting-notes"))
    return reviews_html


def get_variable(variable: str, wine) -> str:
    """Find variable in search result (wine)"""
    result = wine.find(lambda tag: tag.name == "p" and variable in tag.text)
    result = get_text(result)

    if "Rating" in variable:
        return result.replace(variable, "")
    if result:
        return clean_variable(result, variable)
    else:
        return None


def get_text(input):
    return input.text if input else None


def clean_variable(input: str, replaceword: str) -> str:
    """Remove variable word (e.g. "drink_date:") from result"""
    result = input.replace(replaceword, "")  # Remove variable word
    result = re.sub(
        r"[^A-Za-z0-9,. ]+", "", result  # Remove non word characters (enters / slashes etc.)
    ) 
    return result


def get_variables_from_reviews(reviews: list) -> pd.DataFrame:
    """Get variables from reviews and return in a DataFrame"""
    vars = [
        "title",
        "rating",
        "drink_date",
        "reviewed_by",
        "issue_date",
        "source",
        "content",
        "producer",
        "from_location",
        "color",
        "type",
        "sweetness",
        "variety",
    ]
    df = pd.DataFrame(columns=vars)

    for wine in reviews:
        title = get_text(wine.find("h3", class_="titular"))
        rating = get_variable("Rating: ", wine)
        drink_date = get_variable("Drink Date:", wine)
        reviewed_by = get_variable("Reviewed by:", wine)
        issue_date = get_variable("Issue Date:", wine)
        source = get_variable("Source:", wine)
        content = get_text(wine.find("p", class_="tasting-note-content"))
        producer = get_variable("Producer:", wine)
        from_location = get_variable("From:", wine)
        color = get_variable("Color:", wine)
        type = get_variable("Type:", wine)
        sweetness = get_variable("Sweetness:", wine)
        variety = get_variable("Variety:", wine)

        new_wine = {
            "title": title,
            "rating": rating,
            "drink_date": drink_date,
            "reviewed_by": reviewed_by,
            "issue_date": issue_date,
            "source": source,
            "content": content,
            "producer": producer,
            "from_location": from_location,
            "color": color,
            "type": type,
            "sweetness": sweetness,
            "variety": variety,
        }

        df.loc[len(df)] = new_wine
    return df


def export_reviews(country: str, presets: BaseSettings) -> None:
    """Get the reviews from the URL and write to a CSV file"""

    base_url: str = presets.url
    raw_data_dir: Path = presets.raw_data_dir

    logger.info(f"Get wine reviews of {country}")
    driver = log_in_website(driver, base_url)
    page_urls = construct_page_urls(driver, base_url, country)
    reviews = search_reviews(driver, page_urls)
    df = get_variables_from_reviews(reviews)

    logger.info(f"Writing raw data of {country} to {raw_data_dir}")
    file = f"{raw_data_dir}/raw_wines_{country}.csv"
    df.to_csv(file)
