"""Route2 CLI 入口。"""

from __future__ import annotations

from app.route2.cli import parse_args
from app.route2.session_config import Route2SessionConfig
from orchestration.session import Route2LiveSession


def main() -> None:
    args = parse_args()
    cfg = Route2SessionConfig.from_namespace(args)
    Route2LiveSession(cfg).run()


if __name__ == "__main__":
    main()
