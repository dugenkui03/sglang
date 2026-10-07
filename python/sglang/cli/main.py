import argparse

from sglang.cli.utils import get_git_commit_hash
from sglang.version import __version__


def version(args, extra_argv):
    print(f"sglang version: {__version__}")
    print(f"git revision: {get_git_commit_hash()[:7]}")


def main():
    """
    NOTE 类似于这样的服务启动命令入口是这里
        sglang serve \
            --trust-remote-code \                   // 是否可执行模型仓库中的代码
            --model-path Qwen/Qwen3.8-27B-FP8 \     // 模型
            --kv-cache-dtype fp8_e4m3 \             // 保存 KV 用的数据类型
            --mem-fraction-static 0.85 \            // 模型加载和 KV Cache 占用的最大显存比例
            --attention-backend flashinfer \
            --chunked-prefill-size 32768 \          // 组批prefill的时候最多新计算的 token：到达阈值后会把请求分块，会产生 Scheduler.chunked_req
            --max-prefill-tokens 32768 \            // 组批prefill的时候最多计算的 token 数量，达到阈值后讲当前请求整个放进 prefill 批，不会把请求分块
            --reasoning-parser qwen3 \              // 推理结果解析器
            --tool-call-parser qwen3_coder \        // tool-call 结果解析器
            --mamba-full-memory-ratio 4.59 \
            --host 0.0.0.0 \                        // 请求地址
            --port 30000 \
            --mamba-radix-cache-strategy extra_buffer \
            --mamba-ssm-dtype float32
  """
    parser = argparse.ArgumentParser()

    # complex sub commands
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # tip 启动服务
    subparsers.add_parser(
        "serve",
        help="Launch an SGLang server.",
        add_help=False,
    )
    # tip 执行推理
    subparsers.add_parser(
        "generate",
        help="Run inference on a multimodal model.",
        add_help=False,
    )

    # simple commands
    version_parser = subparsers.add_parser(
        "version",
        help="Show the version information.",
    )
    version_parser.set_defaults(func=version)

    args, extra_argv = parser.parse_known_args()

    if args.subcommand == "serve":
        # NOTE 如果是服务启动命令， serve 逻辑分支
        from sglang.cli.serve import serve

        serve(args, extra_argv)
    elif args.subcommand == "generate":
        # NOTE 推理分支
        from sglang.cli.generate import generate

        generate(args, extra_argv)
    elif args.subcommand == "version":
        version(args, extra_argv)
