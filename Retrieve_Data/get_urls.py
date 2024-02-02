import requests
import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
#import pdb

# Base URL for the Yelp search API
base_url = "https://www.yelp.com/search/snippet?find_desc=Restaurants&find_loc=Ann+Arbor%2C+MI&start="
menu_prefix = "https://www.yelp.com/menu/"
start = 0  # Starting page number for the Yelp search results
NUM_PAGES = 24  # Total number of pages to scrape from Yelp search results

# Function to extract restaurant URLs from Yelp search results


def get_urls():
    # pdb.set_trace()
    global start
    with open('restaurant_menu_urls', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Restaurant Name', 'URL'])
        # Loop through pages of Yelp search results
        while start < NUM_PAGES:
            # Construct URL for current page of search results
            search_url = base_url + str(start)

            try:
                search_response = requests.get(search_url)
                # Raise an exception if the response has an error status code
                search_response.raise_for_status()
                # Extract relevant JSON data from response
                search_results = search_response.json(
                )['searchPageProps']['mainContentComponentsListProps']
            except Exception as e:
                print("Error: ", e)
                continue
            # Loop through individual search results
            for result in search_results:
                # Filter for relevant search results
                if result['searchResultLayoutType'] == "iaResult":
                    # Extract restaurant name
                    restaurant_name = result['searchResultBusiness']['name']
                    href = "https://www.yelp.com" + \
                        result['searchResultBusiness']['businessUrl']  # Extract restaurant URL
                    # Filter restaurant URL for menu URL
                    menu_href = filter_for_menus(href)
                    if menu_href:  # If a valid menu URL is found
                        # remove menu prefix - it's used in menu_scraper.py
                        menu_href = menu_href.lstrip(menu_prefix)
                        # Write restaurant name and menu URL to CSV
                        writer.writerow([restaurant_name, menu_href])
            start += 1  # Increment page number for next iteration of loop

# Function to filter restaurant URL for menu URL


def filter_for_menus(href):
    # Replace "biz" with "menu" in URL to construct menu URL
    menu_url = href.replace("biz", "menu")
    # Send HTTP GET request to menu URL
    response = requests.get(menu_url)
    # If a valid page is returned (status code between 200 and 299), return the menu URL
    if response.status_code >= 200 and response.status_code < 300:
        return menu_url
    return None  # If not a valid page, return None


def main():
    get_urls()


if __name__ == "__main__":
    main()
