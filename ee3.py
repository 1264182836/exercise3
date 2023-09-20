import nltk
from nltk.corpus import gutenberg
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# 下载必要的语料库
nltk.download('gutenberg')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# 读取白鲸记文件
whale_text = gutenberg.raw('melville-moby_dick.txt')

# 标记化
tokens = word_tokenize(whale_text)

# 停止词过滤
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

# 词性标记
tagged_tokens = nltk.pos_tag(filtered_tokens)

# 统计词类出现频率
tag_freq = FreqDist(tag for (word, tag) in tagged_tokens)

# 打印五个最常见的词类及其频率
print(tag_freq.most_common(5))

# 词形还原
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens[:20]]

# 获取标签和频率
labels, frequencies = zip(*tag_freq.most_common(30))

# 创建条形图
plt.figure(figsize=(12, 6))
plt.bar(labels, frequencies)
plt.xlabel('tag')
plt.ylabel('frequencies')
plt.xticks(rotation=90)
plt.title('tag_frequencies_distribution')
plt.tight_layout()
plt.show()
