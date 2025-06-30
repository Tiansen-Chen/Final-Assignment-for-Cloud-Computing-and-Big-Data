import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from wordcloud import WordCloud
import matplotlib.colors as mcolors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import requests
import json
import os
from io import BytesIO
import base64

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 确保必要的NLTK数据已下载
# print('start downloading')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# print('done')

# 示例新闻数据
news_data = [
    "The Federal Reserve announced a new policy to combat inflation, raising interest rates by 0.25%. "
    "Economists predict this could slow down economic growth but help stabilize prices in the long run.",

    "Tech giant Apple revealed its latest iPhone model with advanced AI features. "
    "The new device promises better performance and improved camera capabilities, aiming to stay competitive in the smartphone market.",

    "A major breakthrough in cancer research has been reported. Scientists have developed a new immunotherapy "
    "that shows promising results in treating late-stage cancers, offering hope for patients worldwide.",

    "The ongoing climate summit in Paris focuses on reducing global carbon emissions. "
    "World leaders discuss strategies to limit global warming to 1.5 degrees Celsius and transition to renewable energy sources.",

    "Amazon has launched a new drone delivery service in selected cities. "
    "The service aims to deliver packages within 30 minutes, revolutionizing the logistics industry with cutting-edge technology.",

    "A severe heatwave has hit southern Europe, breaking temperature records in many countries. "
    "Authorities warn of health risks and urge citizens to take precautions amid the extreme weather.",

    "Elon Musk's SpaceX successfully launched another batch of Starlink satellites. "
    "The project aims to provide global broadband internet coverage, especially in remote and underserved areas.",

    "The latest unemployment figures show a significant drop, indicating a strong recovery in the job market. "
    "Experts attribute this to increased consumer spending and business expansions post-pandemic.",

    "Researchers have discovered a potential link between excessive screen time and mental health issues in teenagers. "
    "The study recommends balanced digital media use and more outdoor activities for adolescents.",

    "Netflix announced a major investment in original content production. "
    "The streaming giant plans to launch over 500 new shows and movies next year to retain its competitive edge in the market."
]

class NewsTopicAnalyzer:
    def __init__(self, news_data, num_topics=5, ollama_base_url="http://localhost:11434", ollama_model="deepseek-r1"):
        """初始化新闻主题分析器"""
        self.news_data = news_data
        self.num_topics = num_topics
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.output_path = r"C:\Users\86189\Desktop\期末\云计算\two"
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def preprocess_text(self, text):
        """预处理单条新闻文本"""
        # 移除非字母字符并转为小写
        text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
        # 分词
        tokens = text.split()
        # 去停用词和长度≤2的单词
        tokens = [token for token in tokens if token not in self.stopwords and len(token) > 2]
        # 词形还原
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return tokens

    def prepare_data(self):
        """准备文本数据"""
        print("正在预处理新闻数据...")
        self.processed_docs = [self.preprocess_text(doc) for doc in self.news_data]

        # 构建词典
        print("正在构建词典...")
        self.dictionary = corpora.Dictionary(self.processed_docs)
        # 过滤极端词
        self.dictionary.filter_extremes(no_below=1, no_above=0.9)

        # 生成语料库
        print("正在生成语料库...")
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.processed_docs]

        return self

    def train_lda_model(self):
        """训练LDA主题模型"""
        if self.corpus is None:
            self.prepare_data()

        print(f"正在训练LDA模型 (主题数: {self.num_topics})...")
        self.lda_model = models.LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=10,
            alpha='auto',
            eta='auto',
            random_state=42
        )
        print("LDA模型训练完成！")
        return self

    def visualize_topics(self, output_path="lda_visualization.html"):
        """使用pyLDAvis可视化主题模型"""
        if self.lda_model is None:
            self.train_lda_model()

        print("正在生成主题可视化...")
        vis_data = gensimvis.prepare(self.lda_model, self.corpus, self.dictionary)
        full_output_path = os.path.join(self.output_path, output_path)
        pyLDAvis.save_html(vis_data, full_output_path)
        print(f"主题可视化已保存至: {full_output_path}")
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(full_output_path))
            print("已在浏览器中打开可视化页面")
        except Exception as e:
            print(f"无法自动打开浏览器: {e}")
            print(f"请手动在浏览器中打开文件: {full_output_path}")
        else:
            print(f"错误: 文件未生成或路径无效 - {full_output_path}")
        return vis_data

    def generate_wordclouds(self):
        """为每个主题生成词云图"""
        if self.lda_model is None:
            self.train_lda_model()

        print("正在生成主题词云图...")
        # 设置颜色
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

        cloud = WordCloud(
            background_color='white',
            width=800,
            height=600,
            max_words=20,
            colormap='tab10',
            color_func=lambda *args, **kwargs: cols[i],
            prefer_horizontal=1.0
        )

        # 为每个主题生成词云
        for i, topic in enumerate(self.lda_model.show_topics(formatted=False, num_words=20)):
            topic_words = dict(topic[1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)

            plt.figure(figsize=(10, 6))
            plt.imshow(cloud, interpolation="bilinear")
            plt.axis('off')
            plt.title(f'主题 {i+1} 关键词', fontsize=15)
            plt.tight_layout(pad=0)
            full_output_path = os.path.join(self.output_path, f"topic_{i+1}_wordcloud.png")
            plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
            plt.close()

        print(f"已生成 {self.num_topics} 个主题的词云图")

    def call_ollama_api(self, prompt):
        """调用Ollama API获取模型响应"""
        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 500
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            print(f"调用Ollama API失败: {e}")
            return "分析失败，无法连接到Ollama API"

    def analyze_topics_with_ollama(self):
        """使用Ollama部署的DeepSeek-R1模型分析每个主题的内容"""
        if self.lda_model is None:
            self.train_lda_model()

        print("正在使用Ollama部署的DeepSeek-R1模型分析主题内容...")
        topic_analysis = []

        for i, topic in enumerate(self.lda_model.show_topics(formatted=False, num_words=10)):
            topic_words = [word for word, _ in topic[1]]
            topic_bow = self.dictionary.doc2bow(topic_words)
            topic_text = " ".join(topic_words)

            # 构建提示词
            prompt = f"""
            你是一个专业的新闻分析师。以下是从新闻文本中提取的主题关键词：
            "{topic_text}"

            请分析这个主题可能涉及的内容领域，并用简洁的语言概括这个主题。
            请提供：
            1. 主题的简要描述（约20-30字）
            2. 可能的内容领域（如科技、经济、健康等）
            3. 相关的现实场景或事件示例
            """

            # 调用Ollama API
            analysis = self.call_ollama_api(prompt)

            topic_analysis.append({
                "topic_id": i + 1,
                "keywords": topic_words,
                "analysis": analysis
            })

            print(f"\n主题 {i+1} 分析完成:")
            print(f"关键词: {', '.join(topic_words)}")
            print(f"分析结果:\n{analysis}")

        # 保存分析结果
        analysis_df = pd.DataFrame(topic_analysis)
        full_output_path = os.path.join(self.output_path, "topic_analysis.csv")
        analysis_df.to_csv(full_output_path, index=False)
        print(f"主题分析结果已保存至 {full_output_path}")
        return topic_analysis

    def plot_document_topic_heatmap(self):
        """绘制文档-主题概率分布热力图"""
        if self.lda_model is None:
            self.train_lda_model()

        print("正在生成文档-主题热力图...")
        # 获取每个文档的主题分布
        doc_topic_dist = [self.lda_model.get_document_topics(doc, minimum_probability=0) for doc in self.corpus]

        # 转换为矩阵形式
        doc_topic_matrix = np.zeros((len(self.news_data), self.num_topics))
        for doc_idx, topics in enumerate(doc_topic_dist):
            for topic_idx, prob in topics:
                doc_topic_matrix[doc_idx, topic_idx] = prob

        # 创建DataFrame
        topic_names = [f"主题 {i+1}" for i in range(self.num_topics)]
        doc_topic_df = pd.DataFrame(doc_topic_matrix, columns=topic_names)

        # 绘制热力图
        plt.figure(figsize=(12, 8))
        plt.title("文档-主题概率分布热力图", fontsize=15)
        plt.imshow(doc_topic_df, cmap='viridis', aspect='auto')
        plt.colorbar(label='概率')
        plt.xticks(range(self.num_topics), topic_names, rotation=45)
        plt.yticks(range(len(self.news_data)), [f"文档 {i+1}" for i in range(len(self.news_data))])

        # 添加数值标签
        for i in range(len(self.news_data)):
            for j in range(self.num_topics):
                plt.text(j, i, f"{doc_topic_matrix[i, j]:.2f}", 
                         ha="center", va="center", color="w" if doc_topic_matrix[i, j] > 0.5 else "black")

        plt.tight_layout()
        full_output_path = os.path.join(self.output_path, "document_topic_heatmap.png")
        plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"文档-主题热力图已保存至 {full_output_path}")

    def run_analysis(self):
        """运行完整的分析流程"""
        self.prepare_data()
        self.train_lda_model()
        self.visualize_topics()
        self.generate_wordclouds()
        self.plot_document_topic_heatmap()
        self.analyze_topics_with_ollama()

        print("\n===== 分析完成 =====")
        print(f"共处理 {len(self.news_data)} 条新闻")
        print(f"识别出 {self.num_topics} 个主题")
        print(f"结果已保存至 {self.output_path}")

# 主程序
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = NewsTopicAnalyzer(
        news_data=news_data,
        num_topics=5,
        ollama_base_url="http://localhost:11434",  # Ollama API地址
        ollama_model="deepseek-r1"  # Ollama中部署的模型名称
    )

    # 运行完整分析流程
    analyzer.run_analysis()