import os
import tarfile
import urllib

"""Download 150k Python Dataset from SRILAB"""

DATA_DIR = "data"
url = 'http://files.srl.inf.ethz.ch/data/py150.tar.gz'

print("Creating Directory for training and testing data")

if not os.path.exists(DATA_DIR):
    os.mkdir("data")
    print("Directory: './" + DATA_DIR + "' was created ")
else:
    print("Directory: '/" + DATA_DIR + "' already exists ")

print("Downloading Python 150k AST-dataset from: " + url)

file_tmp = urllib.urlretrieve(url, filename=None)[0]
base_name = DATA_DIR 

file_name, file_extension = os.path.splitext(base_name)
tar = tarfile.open(file_tmp)

print("Extracting Files into data directory ")

for member in tar.getmembers():
    filepath = DATA_DIR + "/" + member.name
    if not os.path.exists(filepath):
        tar.extract(member, DATA_DIR)
        print("Extracted File: " + member.name + " into './" + DATA_DIR + "'")
    else:
        print("File: './" + DATA_DIR + '/' + member.name + " already exists. ")
