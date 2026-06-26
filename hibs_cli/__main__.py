"""
Hibs CLI server entry (used by `headless serve`)
=================================================

TypeScript `headless serve` spawns this process:
    python -m hibs_cli serve --ckpt model.pt --port 8000
"""
import argparse
import sys

__version__ = "0.16.6"


def cmd_serve(args):
    """启动 HTTP API 服务"""
    import uvicorn
    from hibs_cli.server import create_app
    app = create_app(args.ckpt, device=args.device)
    print(f"\nAPI server: http://{args.host}:{args.port}")
    print(f"  POST /generate  - text generation")
    print(f"  GET  /info      - model info")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def main():
    parser = argparse.ArgumentParser(prog="hibs", description="Hibs API server")
    parser.add_argument("--version", action="version", version=f"hibs {__version__}")

    sub = parser.add_subparsers(dest="command", help="subcommands")
    p = sub.add_parser("serve", help="Start API server")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
