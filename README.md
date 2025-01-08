# GraphRAG More

GraphRAG More 基于微软 [GraphRAG](https://github.com/microsoft/graphrag) ，支持使用各种大模型：
1. OpenAI接口兼容的模型服务（*微软GraphRAG本就支持，可直接使用微软 [GraphRAG](https://github.com/microsoft/graphrag)* ）
   * OpenAI
   * Azure OpenAI
   * 阿里通义
   * 字节豆包
   * Ollama
   * 其他OpenAI接口兼容的模型服务
2. 非OpenAI接口兼容的模型服务（*微软GraphRAG不支持*）
   * 百度千帆（*推理服务V2版本接口兼容OpenAI，但目前V2版本接口不支持Embedding*）

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

### 1. 安装 GraphRAG More
```shell
pip install graphrag-more

# 如果使用百度千帆，还需要安装qianfan sdk
# pip install qianfan
```

> 如需二次开发或者调试的话，也可以直接使用源码的方式，步骤如下：
>
> **下载 GraphRAG More 代码库**
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
> 如果基于源码方式，请在源码根目录下使用poetry命令运行：
>
> ```shell
> poetry run poe init --root ./ragtest
> ```
这将在./ragtest目录中创建两个文件：`.env`和`settings.yaml`，`.env`包含运行GraphRAG所需的环境变量，`settings.yaml`包含GraphRAG全部设置。

### 4. 配置

**GraphRAG More 1.1.0 版本开始的配置文件与 1.1.0 之前版本的变动较大，升级请注意！！！**

1. `.env`<br/>
在`.env`文件中配置`GRAPHRAG_API_KEY`，这是您所使用的大模型服务的API密钥，将其替换为您自己的API密钥。
   * [阿里通义获取API Key官方文档](https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key)
   * [字节豆包获取API Key 官方文档](https://www.volcengine.com/docs/82379/1361424#%E6%9F%A5%E8%AF%A2-%E8%8E%B7%E5%8F%96-api-key)
   * [百度千帆获取API Key官方文档](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Zm2ycv77m#api_key%E8%AF%B4%E6%98%8E) 
   (注意：百度千帆的`API Key`是带有效期的，过期后需要重新获取)<br/>
   百度千帆还需配置 qianfan sdk 所需的环境变量 `QIANFAN_ACCESS_KEY`、`QIANFAN_SECRET_KEY`，可以配置在系统环境变量中，也可以配置在`.env`文件中，
   参考官方文档：[使用安全认证AK/SK调用流程](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/3lmokh7n6#%E3%80%90%E6%8E%A8%E8%8D%90%E3%80%91%E4%BD%BF%E7%94%A8%E5%AE%89%E5%85%A8%E8%AE%A4%E8%AF%81aksk%E8%B0%83%E7%94%A8%E6%B5%81%E7%A8%8B)
   * Ollama 默认不需要配置`API Key`
2. `settings.yaml`<br/>
在`settings.yaml`文件中，根据您所使用的大模型配置`model`和`api_base`，`GraphRAG More`的`example_settings` 文件夹提供了
百度千帆、阿里通义、字节豆包、Ollama 的`settings.yaml`文件供参考（详细的配置参考微软官方文档：https://microsoft.github.io/graphrag/config/yaml/ ），
根据选用的模型和使用的`GraphRAG More`版本（不同版本`settings.yaml`可能不一样），您可以直接将将`example_settings`
文件夹（比如：`GraphRAG More` 1.1.0 版本的 [example_settings](https://github.com/guoyao/graphrag-more/tree/v1.1.0/example_settings) ）对应模型的`settings.yaml`
文件复制到 ragtest 目录，覆盖初始化过程生成的`settings.yaml`文件。
    ```shell
    # 百度千帆
    cp ./example_settings/qianfan/settings.yaml ./ragtest
    
    # or 阿里通义
    cp ./example_settings/tongyi/settings.yaml ./ragtest
    
    # or 字节豆包
    cp ./example_settings/doubao/settings.yaml ./ragtest
    
    # or ollama
    cp ./example_settings/ollama/settings.yaml ./ragtest
    ```
    `example_settings`的`settings.yaml`里面有的设置了默认的`model`，根据您选用的模型来修改`model`
      * 百度千帆默认使用 ernie-speed-pro-128k 和 tao-8k
      * 阿里通义默认使用 qwen-plus 和 text-embedding-v2
      * 字节豆包需要配置模型ID，即推理接入点ID，不是模型名称
      * Ollama默认使用 mistral:latest 和 quentinz/bge-large-zh-v1.5:latest
        > 对于`Ollama`，需要在构建前安装`Ollama`并下载您选用的模型：
        > * 安装`Ollama`：https://ollama.com/download ，安装后启动
        > * 使用`Ollama`下载模型
        >  ```shell
        >  ollama pull mistral:latest # 默认使用的模型，请替换成您选用的模型
        >  ollama pull quentinz/bge-large-zh-v1.5:latest # 默认使用的模型，请替换成您选用的模型
        >  ```

### 5. 构建索引
```shell
graphrag index --root ./ragtest
```
> 如果基于源码方式，请在源码根目录下使用poetry命令运行：
>
> ```shell
> poetry run poe index --root ./ragtest
> ```
构建过程可能会触发 rate limit （限速）导致构建失败，重复执行几次，或者尝试调小 settings.yaml 中
的 requests_per_minute 和 concurrent_requests 配置，然后重试

### 6. 执行查询
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
> 如果基于源码方式，请在源码根目录下使用poetry命令运行：
>
> ```shell
> # global query
> poetry run poe query \
> --root ./ragtest \
> --method global \
> --query "What are the top themes in this story?"
>
> # local query
> poetry run poe query \
> --root ./ragtest \
> --method local \
> --query "Who is Scrooge, and what are his main relationships?"
> ```
查询过程可能会出现json解析报错问题，原因是某些模型没按要求输出json格式，可以重复执行几次，或者修改 settings.yaml 的 llm.model 改用其他模型

除了使用cli命令之外，也可以使用API方式来查询，以便集成到自己的项目中，API使用方式请参考：
[examples/api_usage](https://github.com/guoyao/graphrag-more/tree/main/examples/api_usage)（注意：不同`GraphRAG More`版本API用法可能不一样，参考所使用版本下的文件）
* 基于已有配置文件查询：[search_by_config_file.py](https://github.com/guoyao/graphrag-more/tree/main/examples/api_usage/search_by_config_file.py)
* 基于代码的自定义查询：[custom_search.py](https://github.com/guoyao/graphrag-more/tree/main/examples/api_usage/custom_search.py)
