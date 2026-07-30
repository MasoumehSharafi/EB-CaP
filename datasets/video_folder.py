from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def natural_key(value: str) -> List[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: natural_key(path.as_posix()),
    )


def resolve_dataset_root(root: str | Path, dataset_dir: str | None) -> Path:
    root_path = Path(root).expanduser().resolve()
    if dataset_dir:
        candidate = root_path / dataset_dir
        if candidate.exists():
            return candidate
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")
    return root_path


def _video_id_from_relative(relative: Path) -> str:
    # Expected layout: <class>/<video>/<frames...>.  If frames are directly in
    # the class folder, treat every frame stem as a one-frame video.
    return relative.parts[0] if len(relative.parts) > 1 else relative.stem


@dataclass(frozen=True)
class VideoRecord:
    subject_id: str
    video_id: str
    class_index: int
    class_name: str
    frames: Tuple[Path, ...]


def collect_source_videos(
    dataset_root: Path,
    split: str,
    class_folders: Sequence[str],
    class_names: Sequence[str],
) -> List[VideoRecord]:
    records: List[VideoRecord] = []
    split_root = dataset_root / split
    for class_index, (folder_name, class_name) in enumerate(zip(class_folders, class_names)):
        class_root = split_root / folder_name
        groups: Dict[str, List[Path]] = {}
        for image_path in list_images(class_root):
            relative = image_path.relative_to(class_root)
            video_id = _video_id_from_relative(relative)
            groups.setdefault(video_id, []).append(image_path)
        for video_id in sorted(groups, key=natural_key):
            frames = tuple(sorted(groups[video_id], key=lambda path: natural_key(path.as_posix())))
            records.append(
                VideoRecord(
                    subject_id="source",
                    video_id=f"{split}/{folder_name}/{video_id}",
                    class_index=class_index,
                    class_name=class_name,
                    frames=frames,
                )
            )
    if not records:
        expected = dataset_root / split / class_folders[0] / "<video>" / "<frame>.jpg"
        raise RuntimeError(f"No source videos found. Expected a layout similar to: {expected}")
    return records


def collect_target_videos(
    dataset_root: Path,
    class_folders: Sequence[str],
    class_names: Sequence[str],
    subject_regex: str = r".+",
    exclude_dirs: Iterable[str] = ("train", "validation", "val", "test"),
) -> List[VideoRecord]:
    pattern = re.compile(subject_regex)
    excluded = set(exclude_dirs)
    records: List[VideoRecord] = []
    subject_dirs = sorted(
        [
            path
            for path in dataset_root.iterdir()
            if path.is_dir()
            and path.name not in excluded
            and not path.name.startswith(".")
            and pattern.fullmatch(path.name)
        ],
        key=lambda path: natural_key(path.name),
    )
    for subject_dir in subject_dirs:
        for class_index, (folder_name, class_name) in enumerate(zip(class_folders, class_names)):
            class_root = subject_dir / folder_name
            groups: Dict[str, List[Path]] = {}
            for image_path in list_images(class_root):
                relative = image_path.relative_to(class_root)
                video_id = _video_id_from_relative(relative)
                groups.setdefault(video_id, []).append(image_path)
            for video_id in sorted(groups, key=natural_key):
                frames = tuple(sorted(groups[video_id], key=lambda path: natural_key(path.as_posix())))
                records.append(
                    VideoRecord(
                        subject_id=subject_dir.name,
                        video_id=f"{folder_name}/{video_id}",
                        class_index=class_index,
                        class_name=class_name,
                        frames=frames,
                    )
                )
    if not records:
        expected = dataset_root / "<subject>" / class_folders[0] / "<video>" / "<frame>.jpg"
        raise RuntimeError(f"No target videos found. Expected a layout similar to: {expected}")
    return records


def sample_clip_indices(num_frames: int, clip_len: int, training: bool, seed_value: int) -> List[int]:
    if num_frames < 1:
        raise ValueError("A video must contain at least one frame.")
    if clip_len < 1:
        raise ValueError("clip_len must be at least 1.")
    if num_frames >= clip_len:
        if training:
            generator = torch.Generator()
            generator.manual_seed(int(seed_value))
            start = int(torch.randint(0, num_frames - clip_len + 1, (1,), generator=generator).item())
            return list(range(start, start + clip_len))
        return torch.linspace(0, num_frames - 1, steps=clip_len).round().long().tolist()
    return list(range(num_frames)) + [num_frames - 1] * (clip_len - num_frames)


def causal_clip_indices(end_index: int, clip_len: int) -> List[int]:
    start = end_index - clip_len + 1
    return [max(0, index) for index in range(start, end_index + 1)]


def load_clip(frames: Sequence[Path], indices: Sequence[int], preprocess: Callable) -> torch.Tensor:
    tensors: List[torch.Tensor] = []
    for index in indices:
        with Image.open(frames[int(index)]) as image:
            tensors.append(preprocess(image.convert("RGB")))
    return torch.stack(tensors, dim=0)


class SourceVideoDataset(Dataset):
    """One or more sampled clips per labeled source video."""

    def __init__(
        self,
        records: Sequence[VideoRecord],
        preprocess: Callable,
        clip_len: int,
        samples_per_video: int = 1,
        training: bool = True,
        seed: int = 0,
    ) -> None:
        if not records:
            raise ValueError("SourceVideoDataset requires at least one video record.")
        self.records = list(records)
        self.preprocess = preprocess
        self.clip_len = int(clip_len)
        self.samples_per_video = max(1, int(samples_per_video))
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records) * self.samples_per_video

    def __getitem__(self, index: int):
        record_index = int(index) // self.samples_per_video
        sample_index = int(index) % self.samples_per_video
        record = self.records[record_index]
        seed_value = self.seed + self.epoch * 1_000_003 + record_index * 997 + sample_index
        indices = sample_clip_indices(len(record.frames), self.clip_len, self.training, seed_value)
        video = load_clip(record.frames, indices, self.preprocess)
        metadata = {"subject_id": record.subject_id, "video_id": record.video_id}
        return video, torch.tensor(record.class_index, dtype=torch.long), metadata


class TargetVideoStreamDataset(Dataset):
    """Causal sliding windows ordered by subject, video, and frame."""

    def __init__(
        self,
        records: Sequence[VideoRecord],
        preprocess: Callable,
        clip_len: int,
        frame_stride: int = 1,
    ) -> None:
        if not records:
            raise ValueError("TargetVideoStreamDataset requires at least one video record.")
        self.records = sorted(
            list(records),
            key=lambda record: (natural_key(record.subject_id), natural_key(record.video_id), record.class_index),
        )
        self.preprocess = preprocess
        self.clip_len = int(clip_len)
        self.frame_stride = max(1, int(frame_stride))
        self.items: List[Tuple[int, int]] = []
        for record_index, record in enumerate(self.records):
            for end_index in range(0, len(record.frames), self.frame_stride):
                self.items.append((record_index, end_index))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        record_index, end_index = self.items[int(index)]
        record = self.records[record_index]
        indices = causal_clip_indices(end_index, self.clip_len)
        video = load_clip(record.frames, indices, self.preprocess)
        metadata = {
            "subject_id": record.subject_id,
            "video_id": record.video_id,
            "frame_index": end_index,
            "frame_path": str(record.frames[end_index]),
        }
        return video, torch.tensor(record.class_index, dtype=torch.long), metadata
