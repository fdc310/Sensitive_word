from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union
import json

# 你的 Aho-Corasick 自动机实现
class TrieNode:
    def __init__(self):
        self.children = {}  # 子节点字典，存储字符到子节点的映射
        self.fail = None    # 失败指针，用于构建 AC 自动机
        self.output = []    # 输出列表，存储以当前节点为结束字符的关键词

class AhoCorasickAutomaton:
    def __init__(self, keywords=None):
        """
        初始化 AC 自动机。
        :param keywords: 可选的关键词列表，用于构建初始的 Trie 树
        """
        self.root = TrieNode()  # 根节点
        if keywords:
            self.build_trie(keywords)  # 如果提供了关键词，则构建 Trie 树

    def build_trie(self, keywords):
        """
        构建 Trie 树。
        :param keywords: 关键词列表
        """
        for keyword in keywords:
            self.add_keyword(keyword)  # 将每个关键词添加到 Trie 树中

    def add_keyword(self, keyword):
        """
        将单个关键词添加到 Trie 树中。
        :param keyword: 关键词字符串
        """
        node = self.root
        for char in keyword:
            if char not in node.children:
                node.children[char] = TrieNode()  # 如果字符不在子节点中，则创建新节点
            node = node.children[char]
        node.output.append(keyword)  # 将关键词添加到当前节点的输出列表中

    def delete_keyword(self, keyword):
        """
        从 Trie 树中删除关键词。
        :param keyword: 要删除的关键词
        """
        def _delete(node, word, index):
            """
            递归删除关键词的辅助函数。
            :param node: 当前节点
            :param word: 要删除的关键词
            :param index: 当前字符的索引
            :return: 是否需要删除当前节点
            """
            if index == len(word):
                if word in node.output:
                    node.output.remove(word)  # 从输出列表中删除关键词
                return not node.output and not node.children  # 如果节点无输出且无子节点，则返回 True
            char = word[index]
            if char in node.children and _delete(node.children[char], word, index + 1):
                del node.children[char]  # 删除子节点
                return not node.output and not node.children
            return False

        _delete(self.root, keyword, 0)
        self.build_fail_links()  # 重新构建失败指针

    def build_fail_links(self):
        """
        构建 AC 自动机的失败指针。
        """
        queue = [self.root]
        while queue:
            current = queue.pop(0)
            for char, child in current.children.items():
                queue.append(child)
                fail_node = current.fail
                while fail_node and char not in fail_node.children:
                    fail_node = fail_node.fail
                child.fail = fail_node.children.get(char) if fail_node else self.root
                child.output += child.fail.output if child.fail else []  # 继承失败节点的输出列表

    def search(self, text):
        """
        使用 AC 自动机搜索文本中的关键词。
        :param text: 待搜索的文本
        :return: 匹配的关键词及其在文本中的起始位置
        """
        node = self.root
        matches = []
        for i, char in enumerate(text):
            while node is not None and char not in node.children:
                node = node.fail  # 沿着失败指针回溯
            node = node.children.get(char, self.root) if node else self.root
            if node.output:
                matches.extend((keyword, i - len(keyword) + 1) for keyword in node.output)
        return matches

# 读取本地词库
def load_keywords_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_keywords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        return text.split(',')


# # 保存本地词库
# def save_keywords(file_path, keywords):
#     with open(file_path, 'w', encoding='utf-8') as file:
#         json.dump(keywords, file, indent=4)
# 保存本地词库
def save_keywords(file_path, keywords):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(f"{','.join(keywords)}")


# 追加关键词
def append_keyword(file_path, keyword):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(f",{','.join(keyword)}")

# 删除关键词
def delete_keyword_from_file(file_path, keyword):
    keywords = load_keywords(file_path)
    if keyword in keywords:
        keywords.remove(keyword)
        # print(keywords)
        save_keywords(file_path, keywords)
        return True
    return False

# 创建 FastAPI 应用
app = FastAPI()

# 定义请求模型
class TextRequest(BaseModel):
    text: str

class KeywordRequest(BaseModel):
    word: str
class KeywordsRequest(BaseModel):
    wordlist: list

class KeyList(BaseModel):
    pagenum: Union[int, None] = 1
    pagesize: int

import time
# 加载关键词
start_time = time.perf_counter()
keywords = load_keywords('keywords.txt')
end_time = time.perf_counter()
average_time = (end_time - start_time)
print(f"执行时间：{average_time}秒")
# print(keywords)
automaton = AhoCorasickAutomaton(keywords)

# 定义替换接口
@app.post("/replace/")
async def replace_text(request: TextRequest):
    text = request.text
    try:
        matches = automaton.search(text)
        # replaced_text = text
        replaced_text = text
        for keyword, pos in matches:
            replaced_text = replaced_text[:pos] + '*' * len(keyword) + replaced_text[pos + len(keyword):]
        return {
            "original_text": text,
            "replaced_text": replaced_text
        }
    except Exception  as e:
        return {"message": "error", "error": str(e)}


# 添加关键词接口
@app.post("/keywords/")
async def add_keyword(request: KeywordsRequest):
    keyword = request.wordlist
    try:
        set_keyword = set(keyword)
        set_keywords = set(keywords)
        unique_keyword = set_keyword-set_keywords
        if len(unique_keyword) < 1:
            raise HTTPException(status_code=400, detail="Keyword already exists")
        # keywords.append(list(tuple(unique_keyword)))
        for i in unique_keyword:
            keywords.append(i)
            # 动态更新自动机
            automaton.add_keyword(i)
        append_keyword('keywords.txt', list(unique_keyword))

        automaton.build_fail_links()  # 重新构建失败指针
        return {"message": "Keyword added successfully"}
    except Exception as e:
        return {"message": "error", "error": str(e)}

@app.post("/del/")
async def delete_keyword(request: KeywordRequest):
    word = request.word
    try:
        # 检查关键词是否存在于列表中
        if not delete_keyword_from_file('keywords.txt', word):
            raise HTTPException(status_code=404, detail="Keyword not found")

            # 更新自动机
        automaton.delete_keyword(word)


        return {"message": "Keyword deleted successfully"}
    except Exception as e:
        return {"message": "error", "error": str(e)}


@app.post("/query/")
async def query_keyword(request: KeyList):
    pagenum = request.pagenum
    pagesize = request.pagesize
    try:
        key_data = keywords[::-1][pagenum-1:pagenum*pagesize]
        count_num = int(len(keywords) / pagenum * pagesize)
        return {"data": {"key_data": key_data, "count_num": count_num, "pagesize": pagesize}}
    except Exception as e:
        return {"message": "error", "error": str(e)}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)