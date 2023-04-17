# Robert Parker wine reviews

### In this project scraped reviews from Robert Parker's website are being analyzed

This repository includes:
- A streamlit dashboard
- Quarto accompanying document
- Scripts to scrape, clean and analyze the reviews

To run the streamlit dashboard:
```streamlit run dashboard.py```


### Instructions
* Run the script without scraping data:
python main.py

* For scraping Robert Parker's reviews: 
    - Create a [Robert Parker](https://www.robertparker.com/) account
    - Create a cred.py in the source folder with your credentials. It should follow this format:
```
username = ""
password = "" 
```
    - Run the script with --country: python main.py --country {country}


For more documentation review the docs directory.
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)