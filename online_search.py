import os
import requests, json
import urllib.parse
from sentence_transformers import SentenceTransformer, util
from multiprocessing.pool import ThreadPool
from PIL import Image
import requests
from io import BytesIO
import torch
import pycurl
import certifi
from bs4 import BeautifulSoup
import re
import pandas as pd
from tqdm import tqdm

print('Loading CLIP Model...')
model = SentenceTransformer('clip-ViT-B-32',device = torch.device("cuda"))

def doImageSearch(full_url):
    returned_code = BytesIO()

    conn = pycurl.Curl()
    conn.setopt(conn.CAINFO, certifi.where())
    conn.setopt(conn.URL, str(full_url))
    conn.setopt(conn.FOLLOWLOCATION, 1)
    conn.setopt(conn.USERAGENT, 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:61.0) Gecko/20100101 Firefox/61.0')
    conn.setopt(conn.WRITEFUNCTION, returned_code.write)
    conn.setopt(conn.WRITEDATA, returned_code)
    conn.perform()
    conn.close()
    return returned_code.getvalue().decode('UTF-8')

def parseResults(code, resized=False):
    """Parse/Scrape the HTML code for the info we want."""
    
    soup = BeautifulSoup(code, 'html.parser')
    with open("result.txt", "w") as o:
        o.write(code)
    results = {
        'links': [],
    }

    # this steps could be refactored to a more compact
    all_script_tags = soup.select('script')

    # # https://regex101.com/r/48UZhY/4
    matched_images_data = ''.join(re.findall(r"AF_initDataCallback\(([^<]+)\);", str(all_script_tags)))
    
    # https://kodlogs.com/34776/json-decoder-jsondecodeerror-expecting-property-name-enclosed-in-double-quotes
    # if you try to json.loads() without json.dumps() it will throw an error:
    # "Expecting property name enclosed in double quotes"
    matched_images_data_fix = json.dumps(matched_images_data)
    matched_images_data_json = json.loads(matched_images_data_fix)

    # https://regex101.com/r/pdZOnW/3
    matched_google_image_data = re.findall(r'\[\"GRID_STATE0\",null,\[\[1,\[0,\".*?\",(.*),\"All\",', matched_images_data_json)

    # https://regex101.com/r/NnRg27/1
    matched_google_images_thumbnails = ', '.join(
        re.findall(r'\[\"(https\:\/\/encrypted-tbn0\.gstatic\.com\/images\?.*?)\",\d+,\d+\]',
                   str(matched_google_image_data))).split(', ')

    for fixed_google_image_thumbnail in matched_google_images_thumbnails:
        # https://stackoverflow.com/a/4004439/15164646 comment by Frédéric Hamidi
        google_image_thumbnail_not_fixed = bytes(fixed_google_image_thumbnail, 'ascii').decode('unicode-escape')
        # after first decoding, Unicode characters are still present. After the second iteration, they were decoded.
        google_image_thumbnail = bytes(google_image_thumbnail_not_fixed, 'ascii').decode('unicode-escape')

    # removing previously matched thumbnails for easier full resolution image matches.
    removed_matched_google_images_thumbnails = re.sub(
        r'\[\"(https\:\/\/encrypted-tbn0\.gstatic\.com\/images\?.*?)\",\d+,\d+\]', '', str(matched_google_image_data))

    # https://regex101.com/r/fXjfb1/4
    # https://stackoverflow.com/a/19821774/15164646
    matched_google_full_resolution_images = re.findall(r"(?:'|,),\[\"(https:|http.*?)\",\d+,\d+\]",
                                                       removed_matched_google_images_thumbnails)
    for index, fixed_full_res_image in enumerate(matched_google_full_resolution_images):
        # https://stackoverflow.com/a/4004439/15164646 comment by Frédéric Hamidi
        original_size_img_not_fixed = bytes(fixed_full_res_image, 'ascii').decode('unicode-escape')
        original_size_img = bytes(original_size_img_not_fixed, 'ascii').decode('unicode-escape')
        results['links'].append(original_size_img)

    # print("Successful search")
    return results

def getImage(url):
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
    }
    try:
        if(url.startswith('https://www.snopes.com')):
            return (False,False)
        response = requests.get(url,timeout=10,headers=headers)
        return (Image.open(BytesIO(response.content)),url)
    except Exception as e_info:
        # print(url)
        # print(e_info)
        return (False,False)

def retrieve(image_path, caption):
    SEARCH_URL = 'https://www.google.com/search?tbm=isch&hl=en&ijn=0&q='

    code = doImageSearch(SEARCH_URL + urllib.parse.quote_plus(caption))

    result = parseResults(code)
    with ThreadPool(16) as pool:
        resource = pool.imap(getImage, result['links'][:20])
        resource_list = list(resource)
        image_list = []
        url_list = []
        for x in resource_list:
            if x[0]:
                image_list.append(x[0])
                url_list.append(x[1])
    if not os.path.exists(image_path):
        image_path = image_path.replace('.jpg','.png')

    if len(image_list) == 0:
        return [False,False]
    embeddings1 = model.encode(Image.open(image_path),batch_size=128, convert_to_tensor=True)
    if len(image_list) == 1:
        embeddings2 = model.encode(image_list[0],batch_size=128, convert_to_tensor=True)
    else:
        embeddings2 = model.encode(image_list,batch_size=128, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    
    # =================
    # NEAR DUPLICATES
    # =================
    # Use a threshold parameter to identify two images as similar. By setting the threshold lower, 
    # you will get larger clusters which have less similar images in it. Threshold 0 - 1.00
    # A threshold of 1.00 means the two images are exactly the same. Since we are finding near 
    # duplicate images, we can set it at 0.99 or any number 0 < X < 1.00.

    return (torch.sort(cosine_scores[0])[0][-1].item(),url_list[torch.sort(cosine_scores[0])[1][-1].item()])