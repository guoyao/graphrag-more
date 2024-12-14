# GraphRAG More

基于 [微软GraphRAG](https://github.com/microsoft/graphrag) ，支持使用百度千帆、阿里通义、Ollama本地模型。

<div align="left">
  <a href="https://pypi.org/project/graphrag-more/">
    <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/graphrag-more">
  </a>
  <a href="https://pypi.org/project/graphrag-more/">
    <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/graphrag-more">
  </a>
</div>

> 可以先熟悉一下微软官方的demo教程：👉 [微软官方文档](https://microsoft.github.io/graphrag/get_started/)


## 使用步骤如下：

要求 [Python 3.10-3.12](https://www.python.org/downloads/)，建议使用 [pyenv](https://github.com/pyenv) 来管理多个python版本

### 1. 安装 graphrag-more
```shell
pip install graphrag-more
```

> 如需二次开发或者调试的话，也可以直接使用源码的方式，步骤如下：
>
> **下载 graphrag-more 代码库**
> ```shell
> git clone https://github.com/guoyao/graphrag-more.git
> ```
>
> **安装依赖包**
> 这里使用 [poetry](https://python-poetry.org) 来管理python虚拟环境
> ```shell
> # 安装 poetry 参考：https://python-poetry.org/docs/#installation
>
> cd graphrag-more
> poetry install
> ```

### 2. 准备demo数据
```shell
# 创建demo目录
mkdir -p ./ragtest/input

# 下载微软官方demo数据
# 微软官方提供的demo数据 https://www.gutenberg.org/cache/epub/24022/pg24022.txt 有点大，会消耗不少token，这里改用精简后的数据
curl https://raw.githubusercontent.com/guoyao/graphrag-more/refs/heads/main/examples/resources/pg24022.txt > ./ragtest/input/book.txt
```

### 3. 初始化demo目录
```shell
graphrag init --root ./ragtest
```

### 4. 移动和修改 settings.yaml 文件
根据选用的模型（千帆、通义、Ollama）和使用的`graphrag-more`版本（不同版本settings.yaml可能不一样），
将 `example_settings` 文件夹（比如：0.5.0.1 版本的[example_settings](https://github.com/guoyao/graphrag-more/tree/v0.5.0.1/example_settings)
）对应模型的 settings.yaml 文件复制到 ragtest 目录，覆盖初始化过程生成的 settings.yaml 文件。
```shell
# 千帆
cp ./example_settings/qianfan/settings.yaml ./ragtest

# or 通义
cp ./example_settings/tongyi/settings.yaml ./ragtest

# or ollama
cp ./example_settings/ollama/settings.yaml ./ragtest
```
每个settings.yaml里面都设置了默认的 llm 和 embeddings 模型，根据选用的模型修改 settings.yaml 文件的 model 配置
* 千帆默认使用 qianfan.ERNIE-Speed-Pro-128K 和 qianfan.tao-8k ，**注意：必须带上 qianfan. 前缀 ！！！**
* 通义默认使用 tongyi.qwen-plus 和 tongyi.text-embedding-v2 ，**注意：必须带上 tongyi. 前缀 ！！！**
* Ollama默认使用 ollama.mistral:latest 和 ollama.quentinz/bge-large-zh-v1.5:latest ，**注意：<=0.3.0版本时，其llm模型不用带前缀，>=0.3.1版本时，其llm模型必须带上 ollama. 前缀，embeddings模型必须带 ollama. 前缀  ！！！**

### 5. 构建前的准备
根据选用的模型，配置对应的环境变量，若使用Ollama需要安装并下载对应模型
* 千帆：需配置环境变量 QIANFAN_AK、QIANFAN_SK（**注意是应用的AK/SK，不是安全认证的Access Key/Secret Key**），如何获取请参考官方文档：https://cloud.baidu.com/doc/WENXINWORKSHOP/s/3lmokh7n6#%E6%AD%A5%E9%AA%A4%E4%B8%80%EF%BC%8C%E8%8E%B7%E5%8F%96%E5%BA%94%E7%94%A8%E7%9A%84ak%E5%92%8Csk
* 通义：需配置环境变量 TONGYI_API_KEY（从0.3.6.1版本开始，也支持使用 DASHSCOPE_API_KEY，同时都配置的情况下 TONGYI_API_KEY 优先级高于 DASHSCOPE_API_KEY），如何获取请参考官方文档：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
* Ollama：
  * 安装：https://ollama.com/download ，安装后启动
  * 下载模型
    ```shell
    ollama pull mistral:latest
    ollama pull quentinz/bge-large-zh-v1.5:latest
    ```

### 6. 构建索引
```shell
graphrag index --root ./ragtest
```
构建过程可能会触发 rate limit （限速）导致构建失败，重复执行几次，或者尝试调小 settings.yaml 中
的 requests_per_minute 和 concurrent_requests 配置，然后重试

### 7. 执行查询
```shell
# global query
graphrag query \
--root ./ragtest \
--method global \
--query "What are the top themes in this story?"

# local query
graphrag query \
--root ./ragtest \
--method local \
--query "Who is Scrooge, and what are his main relationships?"
```
查询过程可能会出现json解析报错问题，原因是某些模型没按要求输出json格式，可以重复执行几次，或者修改 settings.yaml 的 llm.model 改用其他模型

除了使用cli命令之外，也可以使用API方式来查询，以便集成到自己的项目中，API使用方式请参考：
[examples/api_usage](https://github.com/guoyao/graphrag-more/tree/main/examples/api_usage)（注意：不同`graphrag-more`版本API用法可能不一样，参考所使用版本下的文件）
* 基于已有配置文件查询：[search_by_config_file.py](https://github.com/guoyao/graphrag-more/tree/main/examples/api_usage/search_by_config_file.py)
* 基于代码的自定义查询：[custom_search.py](https://github.com/guoyao/graphrag-more/tree/main/examples/api_usage/custom_search.py)
