from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import shutil
import stat
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from src.protocol_runtime import validate_protocol_runtime

import numpy as np

from src.app_manifest import load_app_manifest, protocol_model_dimensions
from src.artifact_compatibility import (
    PUBLIC_ARTIFACT_SCHEMA_VERSION,
    canonical_json_bytes,
    read_regular_file,
    require_secure_artifact_platform,
    sha256_file,
    write_json_atomically,
)
from src.contracts import (
    DEFAULT_DIRICHLET_ALPHA,
    DEFAULT_SPLIT_SEED,
    canonical_client_row_bytes,
    client_shard_metadata,
    dirichlet_split,
)
from src.evaluation_artifact import (
    canonical_source_row_bytes,
    load_scientific_protocol,
    load_evaluation_artifact_snapshot,
    publish_evaluation_artifact,
)
from src.paths import (
    PREPARED_CURRENT_FILENAME,
    PREPARED_GENERATIONS_DIRECTORY,
    PREPARED_GENERATION_SCHEMA_VERSION,
    PREPARED_LEGACY_DIRECTORY,
    RunArtifactLock,
    default_evaluation_artifact_dir,
    default_public_artifact_dir,
    resolve_dir,
    resolve_prepared_artifact_dir,
)
from src.text_preprocessing import create_text_vectorizer


def _preflight_output_root(
    output_dir: str | Path,
    artifact_name: str,
    *,
    reusable: bool,
    allow_prepared_alias: bool = False,
) -> Path:
    """Validate an artifact root without creating or following it.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Configured artifact root.
    artifact_name : str
        Human-readable artifact kind used in errors.
    reusable : bool
        Permit an existing real directory for retryable outputs.
    allow_prepared_alias : bool, optional
        Permit the controlled logical link used by atomic prepared generations.

    Returns
    -------
    pathlib.Path
        Absolute validated output root.

    Raises
    ------
    FileExistsError
        If an immutable output root already exists.
    ValueError
        If the root or its existing parent is a symlink or invalid path type.
    """
    output_path = resolve_dir(output_dir)
    if output_path.is_symlink():
        expected = Path(PREPARED_CURRENT_FILENAME) / artifact_name
        if not (
            allow_prepared_alias
            and reusable
            and Path(os.readlink(output_path)) == expected
        ):
            raise ValueError(f"{artifact_name} artifact root must not be a symlink")
        return output_path
    if output_path.exists():
        if not output_path.is_dir():
            raise ValueError(
                f"{artifact_name} artifact root must be a regular directory"
            )
        if not reusable:
            raise FileExistsError(
                f"{artifact_name} artifact path already exists; use a new path"
            )
    parent = output_path.parent
    if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
        raise ValueError(f"{artifact_name} artifact parent must be a regular directory")
    return output_path


def _validate_output_child(path: Path, parent: Path, *, directory: bool) -> None:
    """Reject unsafe existing publication children before replacement.

    Parameters
    ----------
    path : pathlib.Path
        Direct child output path.
    parent : pathlib.Path
        Validated canonical parent directory.
    directory : bool
        Require a real directory instead of a single-link regular file.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the child escapes, is a link, or has an unsafe file type.
    """
    if path.parent.resolve(strict=True) != parent.resolve(strict=True):
        raise ValueError(f"artifact output escapes its root: {path.name}")
    if not path.exists() and not path.is_symlink():
        return
    try:
        child_stat = path.lstat()
    except OSError as error:
        raise ValueError(f"artifact output is unsafe: {path.name}") from error
    valid = (
        stat.S_ISDIR(child_stat.st_mode)
        if directory
        else stat.S_ISREG(child_stat.st_mode) and child_stat.st_nlink == 1
    )
    if path.is_symlink() or not valid:
        raise ValueError(f"artifact output is unsafe: {path.name}")


def _write_bytes_atomically(path: Path, content: bytes) -> None:
    """Write bytes atomically without following or sharing an existing child.

    Parameters
    ----------
    path : pathlib.Path
        Destination beneath an existing validated directory.
    content : bytes
        Exact bytes to publish.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If an existing destination is not a single-link regular file.
    """
    _validate_output_child(path, path.parent, directory=False)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _validate_client_directory(path: Path, parent: Path) -> None:
    """Validate an existing generated client directory and all direct children.

    Parameters
    ----------
    path : pathlib.Path
        Existing ``client-N`` directory.
    parent : pathlib.Path
        Canonical client artifact root.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the directory or any child is linked or non-regular.
    """
    _validate_output_child(path, parent, directory=True)
    canonical_path = path.resolve(strict=True)
    for child in path.iterdir():
        _validate_output_child(child, canonical_path, directory=False)


def _acquire_preparation_lock(roots: Mapping[str, Path]) -> RunArtifactLock:
    """Acquire a nonblocking exclusive lock for one preparation destination set.

    Parameters
    ----------
    roots : mapping of str to pathlib.Path
        Validated final artifact roots.

    Returns
    -------
    RunArtifactLock
        Held lock that the caller must release.

    Raises
    ------
    RuntimeError
        If another process is preparing the same roots.
    ValueError
        If the lock path is not a single-link regular file.
    """
    material = "\0".join(str(roots[name]) for name in sorted(roots)).encode()
    lock_name = f".fml-prepare-{hashlib.sha256(material).hexdigest()[:16]}.lock"
    lock_path = roots["client"].parent / lock_name
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise ValueError("preparation lock path is unsafe") from error
    file_stat = os.fstat(descriptor)
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        os.close(descriptor)
        raise ValueError("preparation lock path is unsafe")
    file = os.fdopen(descriptor, "a+b")
    try:
        if os.name == "nt":
            import msvcrt

            if file.seek(0, os.SEEK_END) == 0:
                file.write(b"\0")
                file.flush()
            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl

            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        file.close()
        raise RuntimeError(
            "Another artifact preparation is already in progress"
        ) from error
    return RunArtifactLock(lock_path, file)


def _copy_preserved_children(source: Path, staging: Path, owned: set[str]) -> None:
    """Copy allowed non-owned regular files into an owned staging root.

    Parameters
    ----------
    source : pathlib.Path
        Existing reusable output root.
    staging : pathlib.Path
        New owned staging directory.
    owned : set of str
        Generated filenames or prefixes, with prefixes ending in ``*``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If an existing child is linked, non-regular, or otherwise unsafe.
    """
    if not source.exists():
        return
    canonical_source = source.resolve(strict=True)
    for child in source.iterdir():
        is_owned = any(
            child.name.startswith(name[:-1])
            if name.endswith("*")
            else child.name == name
            for name in owned
        )
        if is_owned and child.is_dir():
            _validate_client_directory(child, canonical_source)
        else:
            _validate_output_child(child, canonical_source, directory=False)
        if is_owned:
            continue
        if not child.is_file():
            raise ValueError(
                f"non-owned artifact child must be a regular file: {child.name}"
            )
        _write_bytes_atomically(
            staging / child.name,
            read_regular_file(child, parent=canonical_source),
        )


def _preparation_source(root: Path, artifact_kind: str) -> Path:
    """Select a legacy directory or the active immutable generation as input.

    Parameters
    ----------
    root : pathlib.Path
        Validated logical preparation root.
    artifact_kind : str
        Prepared artifact kind selected below an active generation.

    Returns
    -------
    pathlib.Path
        Existing source directory, or the absent logical root for a first run.

    Raises
    ------
    ValueError
        If an existing logical link does not safely select the active generation.
    """
    if root.is_symlink():
        return resolve_prepared_artifact_dir(root, artifact_kind)
    return root


def _fsync_directory(path: Path) -> None:
    """Flush one directory entry set on platforms that support it.

    Parameters
    ----------
    path : pathlib.Path
        Existing directory to flush.

    Returns
    -------
    None
    """
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_tree(root: Path) -> None:
    """Flush a completed owned artifact tree before final renames.

    Parameters
    ----------
    root : pathlib.Path
        Validated staging root containing only regular files and directories.

    Returns
    -------
    None
    """
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in [*directories, root]:
        _fsync_directory(directory)


def _publish_prepared_roots(
    roots: Mapping[str, Path], stages: Mapping[str, Path]
) -> None:
    """Publish one immutable generation through a single atomic index update.

    Parameters
    ----------
    roots : mapping of str to pathlib.Path
        Logical client, public, and evaluation roots sharing one parent.
    stages : mapping of str to pathlib.Path
        Complete artifact directories within one owned generation staging root.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If roots or stages do not form one contained generation.
    OSError
        If immutable publication or the atomic index update fails.
    """
    if set(roots) != {"client", "public", "evaluation"} or set(stages) != set(roots):
        raise ValueError("prepared publication requires all three artifact kinds")
    parents = {root.parent for root in roots.values()}
    stage_parents = {stage.parent for stage in stages.values()}
    if len(parents) != 1 or len(stage_parents) != 1:
        raise ValueError("prepared roots and stages must each share one parent")
    parent = next(iter(parents))
    generation_stage = next(iter(stage_parents))
    generations = parent / PREPARED_GENERATIONS_DIRECTORY
    if generation_stage.parent != generations:
        raise ValueError("prepared generation staging directory escapes its root")

    generation_id = str(uuid.uuid4())
    generation = generations / generation_id
    write_json_atomically(
        generation_stage / "index.json",
        {
            "schema_version": PREPARED_GENERATION_SCHEMA_VERSION,
            "generation_id": generation_id,
            "logical_roots": {name: roots[name].name for name in sorted(roots)},
        },
        overwrite=False,
    )
    _fsync_directory_tree(generation_stage)
    os.rename(generation_stage, generation)
    _fsync_directory(generations)
    pointer = parent / PREPARED_CURRENT_FILENAME
    if pointer.exists() and not pointer.is_symlink():
        raise ValueError("prepared generation pointer is unsafe")
    legacy_roots = {
        name: root
        for name, root in roots.items()
        if root.exists() and not root.is_symlink()
    }
    for name, root in roots.items():
        expected_alias = Path(PREPARED_CURRENT_FILENAME) / name
        if root.is_symlink() and Path(os.readlink(root)) != expected_alias:
            raise ValueError(f"{name} artifact root has an unsafe prepared alias")

    legacy_archive: Path | None = None
    if legacy_roots:
        legacy_root = parent / PREPARED_LEGACY_DIRECTORY
        if legacy_root.is_symlink() or (
            legacy_root.exists() and not legacy_root.is_dir()
        ):
            raise ValueError("prepared legacy archive must be a regular directory")
        legacy_root.mkdir(mode=0o755, exist_ok=True)
        legacy_archive = legacy_root / generation_id
        legacy_archive.mkdir(mode=0o700)
        _fsync_directory(legacy_root)

    aliases_created: list[Path] = []
    roots_archived: list[tuple[Path, Path]] = []
    temporary_pointer = parent / f".{PREPARED_CURRENT_FILENAME}.{uuid.uuid4().hex}.tmp"
    try:
        for name, root in roots.items():
            if root in legacy_roots.values():
                if legacy_archive is None:
                    raise RuntimeError("legacy archive was not initialized")
                archived = legacy_archive / root.name
                os.rename(root, archived)
                roots_archived.append((root, archived))
            if not root.is_symlink():
                root.symlink_to(
                    Path(PREPARED_CURRENT_FILENAME) / name,
                    target_is_directory=True,
                )
                aliases_created.append(root)
        if legacy_archive is not None:
            _fsync_directory(legacy_archive)
            _fsync_directory(legacy_archive.parent)
            _fsync_directory(parent)
        temporary_pointer.symlink_to(
            Path(PREPARED_GENERATIONS_DIRECTORY) / generation_id,
            target_is_directory=True,
        )
        os.replace(temporary_pointer, pointer)
    except BaseException:
        temporary_pointer.unlink(missing_ok=True)
        for alias in reversed(aliases_created):
            alias.unlink(missing_ok=True)
        for root, archived in reversed(roots_archived):
            os.rename(archived, root)
        if legacy_archive is not None:
            legacy_archive.rmdir()
        raise
    _fsync_directory(parent)


def _raw_split_path(
    dataset_id: str,
    config: str,
    revision: str,
    split: str,
    *,
    local_files_only: bool,
) -> Path:
    """Resolve the immutable raw Parquet file for one dataset split.

    Parameters
    ----------
    dataset_id : str
        Hugging Face dataset repository identifier.
    config : str
        Dataset configuration name.
    revision : str
        Immutable dataset revision.
    split : str
        Split whose Parquet file is required.
    local_files_only : bool
        Require the file to exist in the local cache.

    Returns
    -------
    pathlib.Path
        Resolved cached Parquet path.

    Raises
    ------
    huggingface_hub.errors.HfHubHTTPError
        If the immutable file cannot be resolved from the configured source.
    """
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=dataset_id,
            filename=f"{config}/{split}-00000-of-00001.parquet",
            repo_type="dataset",
            revision=revision,
            local_files_only=local_files_only,
        )
    )


def _validate_loaded_dataset(
    dataset: Any,
    protocol: Mapping[str, Any],
    raw_paths: Mapping[str, Path],
) -> None:
    """Validate loaded dataset identity, rows, labels, and content hashes.

    Parameters
    ----------
    dataset : Any
        Loaded dataset mapping keyed by split name.
    protocol : mapping of str to Any
        Frozen scientific protocol.
    raw_paths : mapping of str to pathlib.Path
        Immutable raw Parquet paths keyed by split name.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any split differs from its frozen identity or checksum contract.
    """
    dataset_spec = protocol["dataset"]
    split_specs = dataset_spec["splits"]
    if set(dataset) != set(split_specs):
        raise ValueError("loaded dataset splits differ from the frozen protocol")

    for split, split_spec in split_specs.items():
        raw_digest = sha256_file(raw_paths[split]).removeprefix("sha256:")
        if raw_digest != split_spec["raw_parquet_sha256"]:
            raise ValueError(f"{split} raw Parquet SHA-256 differs from the protocol")

        loaded_split = dataset[split]
        info = loaded_split.info
        if (
            info.dataset_name != dataset_spec["id"].split("/", maxsplit=1)[1]
            or info.config_name != dataset_spec["config"]
            or loaded_split.column_names != ["text", "label"]
        ):
            raise ValueError("loaded dataset identity differs from the frozen protocol")
        if len(loaded_split) != split_spec["rows"]:
            raise ValueError(f"{split} row count differs from the frozen protocol")

        content_hash = hashlib.sha256()
        label_counts: Counter[int] = Counter()
        label_ranges = split_spec.get("label_index_ranges")
        for index, row in enumerate(loaded_split):
            text = row.get("text")
            label = row.get("label")
            if not isinstance(text, str) or type(label) is not int:
                raise ValueError(f"{split} row {index} has invalid text or label data")
            if label_ranges is not None:
                expected_label = next(
                    (
                        range_label
                        for range_label, (start, end) in enumerate(label_ranges)
                        if start <= index <= end
                    ),
                    None,
                )
                if label != expected_label:
                    raise ValueError(
                        f"{split} labels differ from the frozen official row ranges"
                    )
            content_hash.update(canonical_source_row_bytes(text, label))
            label_counts[label] += 1

        expected_counts = split_spec.get("label_counts")
        if (
            expected_counts is not None
            and [label_counts[label] for label in range(len(expected_counts))]
            != expected_counts
        ):
            raise ValueError(f"{split} label counts differ from the frozen protocol")
        if content_hash.hexdigest() != split_spec["content_sha256"]:
            raise ValueError(
                f"{split} canonical content SHA-256 differs from the frozen protocol"
            )


def load_verified_imdb_dataset(*, local_files_only: bool = False) -> Any:
    """Load and verify every frozen IMDB split before artifact preparation.

    Parameters
    ----------
    local_files_only : bool, optional
        Require all dataset metadata and bytes to be available in local caches.

    Returns
    -------
    datasets.DatasetDict
        Verified official train, test, and unsupervised splits.

    Raises
    ------
    ValueError
        If package version, identity, rows, labels, or checksums differ from the
        frozen scientific protocol.
    """
    from datasets import DownloadConfig, load_dataset

    protocol = load_scientific_protocol()
    dataset_spec = protocol["dataset"]
    installed_version = importlib.metadata.version("datasets")
    if installed_version != dataset_spec["datasets_version"]:
        raise ValueError(
            "datasets version differs from the frozen protocol: "
            f"expected {dataset_spec['datasets_version']}, got {installed_version}"
        )

    raw_paths = {
        split: _raw_split_path(
            dataset_spec["id"],
            dataset_spec["config"],
            dataset_spec["revision"],
            split,
            local_files_only=local_files_only,
        )
        for split in dataset_spec["splits"]
    }
    dataset = load_dataset(
        dataset_spec["id"],
        dataset_spec["config"],
        revision=dataset_spec["revision"],
        download_config=DownloadConfig(local_files_only=local_files_only),
    )
    _validate_loaded_dataset(dataset, protocol, raw_paths)
    return dataset


def _validated_frameworks() -> tuple[Any, Any, Mapping[str, Any]]:
    """Load frameworks only when their versions match the frozen protocol.

    Returns
    -------
    tuple
        Keras, TensorFlow, and the parsed frozen protocol.

    Raises
    ------
    ValueError
        If any registered framework version differs from the runtime.
    """
    import keras
    import tensorflow as tf

    protocol = validate_protocol_runtime()
    return keras, tf, protocol


def build_vectorizer(texts: Any) -> Any:
    """Build and verify the shared public token-ID contract.

    Parameters
    ----------
    texts : Any
        Complete official training texts in ascending row-index order.

    Returns
    -------
    keras.layers.TextVectorization
        Adapted vectorizer matching the frozen vocabulary contract.
    """
    _, tf, protocol = _validated_frameworks()
    preprocessing = protocol["preprocessing"]
    dimensions = protocol_model_dimensions(protocol)
    vectorizer = create_text_vectorizer(
        sequence_length=dimensions["sequence_length"],
        max_tokens=preprocessing["max_tokens"],
    )
    vectorizer.adapt(
        tf.data.Dataset.from_tensor_slices(list(texts)).batch(256, drop_remainder=False)
    )
    vocabulary = vectorizer.get_vocabulary()
    vocabulary_bytes = b"".join(item.encode("utf-8") + b"\n" for item in vocabulary)
    if len(vocabulary) != preprocessing["vocabulary_size"]:
        raise ValueError("vocabulary size differs from the frozen protocol")
    if (
        hashlib.sha256(vocabulary_bytes).hexdigest()
        != preprocessing["vocabulary_sha256"]
    ):
        raise ValueError("vocabulary SHA-256 differs from the frozen protocol")
    return vectorizer


def package_raw_client_shards(
    texts: Any,
    labels: Any,
    output_dir: str | Path,
    num_clients: int,
    *,
    alpha: float = DEFAULT_DIRICHLET_ALPHA,
    seed: int = DEFAULT_SPLIT_SEED,
    manifest: Mapping[str, Any],
    dataset: Mapping[str, Any],
    source_split: str = "train",
) -> list[Path]:
    """Write raw review text and labels into one directory per client.

    Parameters
    ----------
    texts : Any
        Review texts to partition.
    labels : Any
        Labels corresponding to ``texts``.
    output_dir : str or pathlib.Path
        Parent directory for client shards.
    num_clients : int
        Number of client shards to create.
    alpha : float, optional
        Dirichlet concentration parameter.
    seed : int, optional
        Random seed for deterministic partitioning.
    manifest : mapping
        Canonical public manifest bound into every shard.
    dataset : mapping
        Frozen official train-dataset identity bound into every shard.
    source_split : str, optional
        Official split name used to qualify stable row identities.

    Returns
    -------
    list of pathlib.Path
        Created client shard directories.
    """
    text_array = np.asarray(texts)
    label_array = np.asarray(labels)
    if len(text_array) != len(label_array):
        raise ValueError(
            "text/label sample count mismatch: "
            f"len(texts)={len(text_array)} len(labels)={len(label_array)}"
        )
    if source_split != "train" or any(
        type(label) is not int or label not in (0, 1) for label in label_array.tolist()
    ):
        raise ValueError("client shards require binary labels from the train split")

    output_path = _preflight_output_root(output_dir, "client", reusable=True)
    output_path.mkdir(parents=True, exist_ok=True)
    canonical_output = output_path.resolve(strict=True)
    for child in output_path.iterdir():
        if child.name.startswith("client-"):
            if child.is_dir():
                _validate_client_directory(child, canonical_output)
            else:
                _validate_output_child(child, canonical_output, directory=False)
    manifest_bytes = canonical_json_bytes(manifest)

    split = dirichlet_split(
        label_array,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
    )

    staging_root = Path(
        tempfile.mkdtemp(dir=output_path, prefix=".shards.", suffix=".tmp")
    )
    try:
        for client_id in range(num_clients):
            indices = split[client_id]
            client_labels = label_array[indices]
            records_bytes = b"".join(
                canonical_client_row_bytes(
                    f"{source_split}:{source_index}",
                    str(text_array[source_index]),
                    int(label),
                )
                for source_index, label in zip(indices, client_labels, strict=True)
            )
            shard_dir = staging_root / f"client-{client_id}"
            shard_dir.mkdir()
            _write_bytes_atomically(shard_dir / "reviews.jsonl", records_bytes)
            shard_metadata = client_shard_metadata(
                client_id,
                client_labels,
                records_bytes=records_bytes,
                public_manifest_bytes=manifest_bytes,
                dataset=dataset,
                source_split=source_split,
                split_seed=seed,
                alpha=alpha,
            )
            write_json_atomically(shard_dir / "client_metadata.json", shard_metadata)

        for child in list(output_path.iterdir()):
            if child.name.startswith("client-"):
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        shard_paths = []
        for client_id in range(num_clients):
            shard_dir = output_path / f"client-{client_id}"
            os.rename(staging_root / shard_dir.name, shard_dir)
            shard_paths.append(shard_dir)
        return shard_paths
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def publish_public_artifacts(
    vectorizer: Any,
    output_dir: str | Path,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish the vocabulary and its train-only provenance contract.

    Parameters
    ----------
    vectorizer : Any
        Verified training-split vectorizer.
    output_dir : str or pathlib.Path
        Public artifact directory.
    protocol : mapping or None, optional
        Parsed frozen protocol, primarily for deterministic tests.

    Returns
    -------
    dict of str to Any
        Published public manifest.
    """
    output_path = _preflight_output_root(output_dir, "public", reusable=True)
    output_path.mkdir(parents=True, exist_ok=True)
    vocabulary = vectorizer.get_vocabulary()
    vocabulary_bytes = b"".join(item.encode("utf-8") + b"\n" for item in vocabulary)
    frozen = protocol or load_scientific_protocol()
    preprocessing = frozen["preprocessing"]
    dimensions = protocol_model_dimensions(frozen)
    vocabulary_sha256 = hashlib.sha256(vocabulary_bytes).hexdigest()
    if len(vocabulary) != preprocessing["vocabulary_size"]:
        raise ValueError("vocabulary size differs from the frozen protocol")
    if vocabulary_sha256 != preprocessing["vocabulary_sha256"]:
        raise ValueError("vocabulary SHA-256 differs from the frozen protocol")
    canonical_output = output_path.resolve(strict=True)
    for filename in ("vocab.txt", "manifest.json"):
        _validate_output_child(
            output_path / filename, canonical_output, directory=False
        )
    _write_bytes_atomically(output_path / "vocab.txt", vocabulary_bytes)
    dataset = frozen["dataset"]
    train = dataset["splits"]["train"]
    manifest = {
        "schema_version": PUBLIC_ARTIFACT_SCHEMA_VERSION,
        **dimensions,
        "vocabulary": {
            "filename": "vocab.txt",
            "sha256": vocabulary_sha256,
            "size_bytes": len(vocabulary_bytes),
        },
        "dataset": {
            "id": dataset["id"],
            "config": dataset["config"],
            "revision": dataset["revision"],
            "datasets_version": dataset["datasets_version"],
            "split": "train",
            "rows": train["rows"],
            "raw_parquet_sha256": train["raw_parquet_sha256"],
            "content_sha256": train["content_sha256"],
        },
    }
    write_json_atomically(output_path / "manifest.json", manifest)
    return manifest


def prepare_all(
    partitions: int,
    client_shard_dir: str | Path,
    public_artifact_dir: str | Path,
    evaluation_artifact_dir: str | Path,
) -> None:
    """Create train-only client/public artifacts and the separate test artifact.

    Parameters
    ----------
    partitions : int
        Number of client training shards.
    client_shard_dir : str or pathlib.Path
        Parent directory for train-only client shards.
    public_artifact_dir : str or pathlib.Path
        Directory for the shared train-derived vocabulary.
    evaluation_artifact_dir : str or pathlib.Path
        New immutable directory for the untouched official test split.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If direct preparation runs outside the supported Linux platform.
    """
    require_secure_artifact_platform()
    client_dir = _preflight_output_root(
        client_shard_dir, "client", reusable=True, allow_prepared_alias=True
    )
    public_dir = _preflight_output_root(
        public_artifact_dir, "public", reusable=True, allow_prepared_alias=True
    )
    evaluation_dir = _preflight_output_root(
        evaluation_artifact_dir,
        "evaluation",
        reusable=True,
        allow_prepared_alias=True,
    )
    roots = {
        "client": client_dir,
        "public": public_dir,
        "evaluation": evaluation_dir,
    }
    for first_name, first in roots.items():
        for second_name, second in roots.items():
            if first_name >= second_name:
                continue
            if (
                first == second
                or first.is_relative_to(second)
                or second.is_relative_to(first)
            ):
                raise ValueError(
                    f"{first_name} and {second_name} artifact roots must be separate"
                )
    for root in roots.values():
        root.parent.mkdir(parents=True, exist_ok=True)
    if len({root.parent for root in roots.values()}) != 1:
        raise ValueError(
            "client, public, and evaluation artifact roots must share one parent for "
            "atomic generation publication"
        )
    lock = _acquire_preparation_lock(roots)
    stages: dict[str, Path] = {}
    try:
        parent = next(iter({root.parent for root in roots.values()}))
        generations = parent / PREPARED_GENERATIONS_DIRECTORY
        if generations.is_symlink() or (
            generations.exists() and not generations.is_dir()
        ):
            raise ValueError("prepared generations root must be a regular directory")
        generations.mkdir(mode=0o755, exist_ok=True)
        for residue in generations.glob(".prepare-*.staging"):
            residue_stat = residue.lstat()
            if residue.is_symlink() or not stat.S_ISDIR(residue_stat.st_mode):
                raise ValueError(f"owned preparation residue is unsafe: {residue.name}")
            shutil.rmtree(residue)
        generation_stage = Path(
            tempfile.mkdtemp(dir=generations, prefix=".prepare-", suffix=".staging")
        )
        stages = {name: generation_stage / name for name in roots}
        stages["public"].mkdir()
        stages["client"].mkdir()
        public_source = _preparation_source(public_dir, "public")
        client_source = _preparation_source(client_dir, "client")
        _copy_preserved_children(
            public_source, stages["public"], {"manifest.json", "vocab.txt"}
        )
        _copy_preserved_children(client_source, stages["client"], {"client-*"})

        _validated_frameworks()
        dataset = load_verified_imdb_dataset()
        train = dataset["train"]
        texts = np.asarray(train["text"])
        labels = np.asarray(train["label"], dtype="int32")
        vectorizer = build_vectorizer(texts)
        protocol = load_scientific_protocol()
        manifest = publish_public_artifacts(
            vectorizer, stages["public"], protocol=protocol
        )
        package_raw_client_shards(
            texts,
            labels,
            stages["client"],
            num_clients=partitions,
            manifest=manifest,
            dataset=manifest["dataset"],
        )
        publish_evaluation_artifact(
            dataset["test"], stages["evaluation"], protocol=protocol
        )

        app_manifest = load_app_manifest(
            public_artifact_dir=stages["public"], protocol=protocol
        )
        identities: set[str] = set()
        for client_id in range(partitions):
            from src.local_training import load_client_shard_snapshot

            snapshot = load_client_shard_snapshot(
                stages["client"] / f"client-{client_id}",
                app_manifest,
                client_id,
            )
            shard_identities = {row[0] for row in snapshot.rows}
            if identities & shard_identities:
                raise ValueError("client shard row identities overlap")
            identities.update(shard_identities)
        if identities != {
            f"train:{index}" for index in range(manifest["dataset"]["rows"])
        }:
            raise ValueError("client shards do not exactly partition the train split")
        load_evaluation_artifact_snapshot(stages["evaluation"], protocol=protocol)
        for stage in stages.values():
            _fsync_directory_tree(stage)
        _publish_prepared_roots(roots, stages)
        stages = {}
    finally:
        if stages:
            shutil.rmtree(next(iter(stages.values())).parent, ignore_errors=True)
        lock.release()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse data-preparation command-line arguments.

    Parameters
    ----------
    argv : list of str or None, optional
        Explicit arguments, or process arguments when omitted.

    Returns
    -------
    argparse.Namespace
        Validated preparation options.
    """
    parser = argparse.ArgumentParser(description="Prepare sentiment data partitions.")
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument(
        "--client-shard-dir", type=Path, default=Path("artifacts/clients")
    )
    parser.add_argument(
        "--public-artifact-dir", type=Path, default=default_public_artifact_dir()
    )
    parser.add_argument(
        "--evaluation-artifact-dir",
        type=Path,
        default=default_evaluation_artifact_dir(),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run verified train/test artifact preparation.

    Parameters
    ----------
    argv : list of str or None, optional
        Explicit command-line arguments, or process arguments when omitted.

    Returns
    -------
    None
    """
    args = parse_args(argv)
    prepare_all(
        partitions=args.partitions,
        client_shard_dir=args.client_shard_dir,
        public_artifact_dir=args.public_artifact_dir,
        evaluation_artifact_dir=args.evaluation_artifact_dir,
    )


if __name__ == "__main__":
    main()
