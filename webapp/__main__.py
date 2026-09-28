# -*- coding: utf-8 -*-
"""
Запуск веб-приложения: python -m webapp [--port 8050] [--open].

Сервер слушает только 127.0.0.1: приложение работает с выгрузками заказчика
и не предназначено для доступа из сети.
"""

import argparse
import threading
import webbrowser


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Smart Grid: веб-приложение")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--open", action="store_true", help="Открыть браузер после запуска")
    args = parser.parse_args(argv)

    import uvicorn
    from webapp.app import create_app

    url = f"http://127.0.0.1:{args.port}/"
    if args.open:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"Smart Grid: {url}  (остановить — Ctrl+C)")
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
