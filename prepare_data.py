import json
import os
import pickle
import sys
import time
import numpy as np
from utils import add_special_tokens

#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
dm_single_close_quote = '\u2019' # unicode
dm_double_close_quote = '\u201d'
# acceptable ways to end a sentence
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"',
              dm_single_close_quote, dm_double_close_quote, ")"]


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line:
        return line
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + " ."


def get_art_abs(lines):
    """ return as list of sentences"""

    # truncated trailing spaces, and normalize spaces
    lines = [' '.join(line.strip().split()) for line in lines]
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)
    return ' '.join(article_lines), ' '.join(highlights)


def write_json(i,article, abstract,size_max):
	""" Saves a json file."""

	file = "./gpt2_"+str(size_max)+"_data/"+str(i)+".json"
	js_example = {}
	js_example['id'] = i
	js_example['article'] = article
	js_example['abstract'] = abstract
	with open(file, 'w') as f:
		json.dump(js_example, f, ensure_ascii=False)


def main(file_names, directory, size_max):
	""" Reads txt files, extract articles and summaries, tokenize them and save as json files
		Args:
			file_names: list, all the articles with total no of tokens less than size_max
			directory: string, directory where files in file_names is stored
	"""
	tokenizer = add_special_tokens()
	print("Execution Started...")
	train_ids = []
	file_id_map = {}
	i = 0
	for file in file_names:
		print(file)
		file = os.path.join(os.getcwd(),directory,file)
		with open(file,'r',encoding='utf-8') as f:
			lines = f.read().split('\n\n')
		article, abstract = get_art_abs(lines)
		article, abstract = tokenizer.encode(article), tokenizer.encode(abstract)
		if len(article)>0 and len(abstract)>0 and (len(article)+len(abstract))<=1023:
			train_ids.append(i)
			write_json(i,article,abstract,size_max)
			file_id_map[i] = os.path.basename(file).replace('.story', '')
			i += 1
			if i%100==0:
				print(i, " files written")


	x,y = int(len(train_ids)*0.8), int(len(train_ids)*0.9)
	valid_ids = train_ids[x:y]
	test_ids = train_ids[y:]
	train_ids = train_ids[:x]
	with open("ids.json",'w') as f:
		js = {}
		js['train_ids'] = train_ids
		js['valid_ids'] = valid_ids
		js['test_ids'] = test_ids
		json.dump(js,f)

	# file_id_map maps the json file ids to actual cnn/dm file names ending with ".story"
	print("saving file_id_map...")
	with open("file_id_map.pickle", 'wb') as f:
		pickle.dump(file_id_map,f)
	print("file_id_map saved.")

def main_less(file_names, directory, size_max, train_size=4096, valid_size=500, test_size=500):
    tokenizer = add_special_tokens()
    print("Execution Started...")
    all_ids = []
    i = 0
    for file in file_names:
        print(file)
        file = os.path.join(os.getcwd(), directory, file)
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n\n')
        article, abstract = get_art_abs(lines)
        article, abstract = tokenizer.encode(article), tokenizer.encode(abstract)
        if len(article) > 0 and len(abstract) > 0 and (len(article) + len(abstract)) <= 1023:
            all_ids.append(i)
            write_json(i, article, abstract, size_max)
            i += 1

    # Randomly shuffle all_ids and then split into train, valid, test
    np.random.shuffle(all_ids)
    train_ids = all_ids[:train_size]
    valid_ids = all_ids[train_size:train_size + valid_size]
    test_ids = all_ids[train_size + valid_size:train_size + valid_size + test_size]

    with open("ids.json", 'w') as f:
        js = {"train_ids": train_ids, "valid_ids": valid_ids, "test_ids": test_ids}
        json.dump(js, f)

    print("Data split: {} training, {} validation, {} testing.".format(len(train_ids), len(valid_ids), len(test_ids)))
    
if __name__ == '__main__':
	size_max = 1024
	start = time.time()
	with open(sys.argv[1],'rb') as f:
		file_sizes = pickle.load(f)
	file_names = [file for file,size in file_sizes.items() if size<=size_max] #only consider files with total no of tokens less than size_max
	if "cnn" in sys.argv[1].lower():
		print("start check cnn files")
		directory = "data\cnn_stories\cnn\stories"
	elif "dm" in sys.argv[1].lower():
		print("start check dailymail files")
		directory = "data\dailymail_stories\dailymail\stories"
	else:
		raise KeyError('error')
	main_less(file_names, directory, size_max)
	print("total_time_taken: ", (time.time()-start)/60, " minutes")