# GraphRAG More

基于 [微软GraphRAG](https://github.com/microsoft/graphrag) ，支持使用百度千帆、阿里通义、Ollama本地模型。

👉 [官方文档](https://microsoft.github.io/graphrag/posts/get_started/)

## 使用步骤如下：

### 1. 下载 graphrag-more 代码库
```shell
git clone https://github.com/guoyao/graphrag-more.git
```

### 2. 准备Python虚拟环境
要求 [Python 3.10-3.12](https://www.python.org/downloads/)，如果使用系统全局python，
可以跳过这一步，不过强烈建议使用 pyenv 或者 conda 来管理虚拟环境

使用 pyenv 参考如下：
```shell
# 安装和配置 pyenv
# 参考：https://github.com/pyenv/pyenv

# 安装和配置 pyenv-virtualenv 插件
# 参考：https://github.com/pyenv/pyenv-virtualenv

# 安装python并创建虚拟环境
pyenv install 3.10.14
pyenv virtualenv 3.10.14 graphrag-more

# 激活虚拟环境
cd graphrag-more # cd到项目根目录
python activate graphrag-more
```

### 3. 安装依赖包
```shell
cd graphrag-more
pip install -r requirements.txt
```

### 4. 准备demo数据
```shell
# 创建demo目录
mkdir -p ./ragtest/input

# 下载微软官方demo数据
curl https://www.gutenberg.org/cache/epub/24022/pg24022.txt > ./ragtest/input/book.txt
```

### 5. 初始化demo目录
```shell
python -m graphrag.index --init --root ./ragtest
```

### 6. 移动和修改 settings.yaml 文件
根据选用的模型（千帆、通义、Ollama）将 example_settings 文件夹对应模型的 settings.yaml 文件复制到 ragtest 目录，
覆盖初始化过程生成的 settings.yaml 文件。
```shell
# 千帆
cp ./example_settings/qianfan/settings.yaml ./ragtest

# or 通义
cp ./example_settings/tongyi/settings.yaml ./ragtest

# or ollama
cp ./example_settings/ollama/settings.yaml ./ragtest
```
每个settings.yaml里面都设置了默认的 llm 和 embeddings 模型，根据你自己要使用的模型修改 settings.yaml 文件的 model 配置
* 千帆默认使用 qianfan.ERNIE-4.0-Turbo-8K 和 qianfan.bge-large-zh ，**注意：必须带上 qianfan 前缀 ！！！**
* 通义默认使用 tongyi.qwen-plus 和 tongyi.text-embedding-v2 ，**注意：必须带上 tongyi 前缀 ！！！**
* Ollama默认使用 mistral:latest 和 ollama.quentinz/bge-large-zh-v1.5:latest，**注意：其llm模型不用带前缀，embeddings模型必须带 ollama 前缀  ！！！**

### 7. 构建前的准备
根据选用的模型，配置对应的环境变量，若使用Ollama需要安装并下载对应模型
* 千帆：需配置环境变量 QIANFAN_AK、QIANFAN_SK ，如何获取请参考官方文档
* 通义：需配置环境变量 TONGYI_API_KEY ，如何获取请参考官方文档
* Ollama：（类似Docker）
  * 安装：https://ollama.com/download ，安装后启动
  * 下载模型
    ```shell
    ollama pull mistral:latest
    ollama pull quentinz/bge-large-zh-v1.5:latest
    ```

### 8. 构建索引
```shell
python -m graphrag.index --root ./ragtest
```
构建过程可能会触发 rate limit （限速）导致构建失败，重复执行几次，或者尝试调小 settings.yaml 中
的 requests_per_minute 和 concurrent_requests 配置，然后重试

### 9. 执行查询
```shell
# global query
python -m graphrag.query \
--root ./ragtest \
--method global \
"What are the top themes in this story?"

# local query
python -m graphrag.query \
--root ./ragtest \
--method local \
"Who is Scrooge, and what are his main relationships?"
```
查询过程可能会出现json解析报错问题，原因是某些模型没按要求输出json格式，可以重复执行几次，或者修改 settings.yaml 的 llm.model 改用其他模型

