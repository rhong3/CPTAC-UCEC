from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import os

def main():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1024x1400")

    # download Chrome Webdriver
    # https://sites.google.com/a/chromium.org/chromedriver/download
    # put driver executable file in the script directory
    chrome_driver = os.path.join(os.getcwd(), "chromedriver")

    driver = webdriver.Chrome(chrome_options=chrome_options, executable_path=chrome_driver)
    driver.get('https://pathology.cancerimagingarchive.net/pathdata/cptac_camicroscope/osdCamicroscope.php?tissueId=C3N-02436-27')

    htmlSource = driver.page_source
    print(htmlSource)


if __name__ == '__main__' : main()


# https://pathology.cancerimagingarchive.net/pathdata/data/cptac/images/batch5/111655.svs
# https://pathology.cancerimagingarchive.net/pathdata/data/cptac/images/batch5/115221.svs
# <a href="/pathdata/data/cptac/images/batch5/115221.svs">