from pathlib import Path, PurePath
from typing import List


def path2str(path: Path) -> str:
    return str(path.resolve())


def read_from_file(path: str, parent: str = "") -> List[str]:
    content = []
    with Path(path).open("r") as f:
        content = f.readlines()
    content = [
        str(PurePath(parent).joinpath(line.strip()))
        for line in content if len(line.strip()) > 0
    ]
    return content
