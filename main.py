
def load_keywords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        return text.split(',')


load_keywords("keywords.txt")