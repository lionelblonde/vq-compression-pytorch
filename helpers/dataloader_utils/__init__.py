from pathlib import Path, PurePath
from typing import List


def get_num_samples(filename: str) -> int:
    with open(filename, "r") as f:
        lines = f.readlines()
    num_samples = len([line for line in lines if line.strip()])
    return num_samples


def path2str(path: Path) -> str:
    return str(path.resolve())


def read_from_file(path: str, parent: str = "") -> List[str]:
    content = []
    with open(path, "r") as f:
        content = f.readlines()
    content = [str(PurePath(parent).joinpath(line.strip())) for line in content if len(line.strip()) > 0]
    return content


def read_path_from_file(path: str, parent: str) -> List[Path]:
    content = read_from_file(path, parent)
    content = [Path(line) for line in content]
    return content


def read_list_from_file(path: str) -> List[List[int]]:
    content = read_from_file(path)
    content = [[int(li.rstrip()) for li in line[1:-1].split(',')] for line in content]
    return content


def save2file(filepath: Path, content: List[str]) -> None:
    with open(path2str(filepath), "w") as f:
        for line in content:
            f.write(f"{line}\n")
