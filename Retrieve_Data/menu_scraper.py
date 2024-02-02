from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import re
import csv
import pandas as pd


class MenuScraper:

    global menu_prefix

    menu_prefix = "https://www.yelp.com/menu/"

    def create_dictionary(self):
        restaurant_url = dict()
        with open("restaurat_menu_urls.csv") as f:
            lines = f.readlines()
            lines.pop(0)
            for l in lines:
                name = (l.split(",")[0]).strip()
                url = (l.split(",")[1]).strip()
                restaurant_url[name] = menu_prefix + url
        return restaurant_url

    def remove_all_tags(self, mystr):
        return re.compile(r'<[^>]+>').sub('', mystr).strip()

    def scrape_menus(self):
        # driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver") DEPRECATED LINE
        menu_items = []
        restaurant_url = self.create_dictionary()
        restaurant_menus = dict()
        driver = webdriver.Chrome(service=Service(
            ChromeDriverManager().install()))

        for k in restaurant_url.keys():
            url = restaurant_url[k]
            driver.get(url)
            content = driver.page_source
            soup = BeautifulSoup(content, features="html.parser")
            menuitems = soup.find_all("div", attrs={"class": [
                                      "arrange_unit arrange_unit--fill menu-item-details menu-item-no-photo", "arrange_unit arrange_unit--fill menu-item-details"]})
            for item in menuitems:
                myitem = str(item.find("h4"))
                itemdetails = str(item.find("p"))
                menu_items.append((self.remove_all_tags(
                    myitem), self.remove_all_tags(itemdetails)))

            restaurant_menus[k] = (menu_items)
            menu_items = []
        print(restaurant_menus)
        return restaurant_menus


def main():
    s = MenuScraper()
    mydict = s.scrape_menus()

    # Write output to a CSV file
    with open("menu_data.csv", "w", newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['Restaurant Name', 'Menu Item', 'Item Description'])
        for s in mydict.keys():
            for pair in mydict[s]:
                writer.writerow([s, pair[0], pair[1]])


if __name__ == "__main__":
    main()
