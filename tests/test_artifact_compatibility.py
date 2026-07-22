import errno
import hashlib
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from src.app_manifest import load_app_manifest
from src.artifact_compatibility import (
    ARTIFACT_SCHEMA_VERSION,
    PUBLIC_ARTIFACT_SCHEMA_VERSION,
    SERVER_ARTIFACT_SCHEMA_VERSION,
    SERVER_ARTIFACTS,
    RetainedDirectoryChain,
    canonical_json_bytes,
    load_server_artifact_manifest,
    load_server_artifact_snapshot,
    read_regular_file,
    require_secure_artifact_platform,
    strict_json_loads,
    validate_artifact_schema,
    write_bytes_atomically,
    write_server_artifact_manifest,
)
from tests.artifact_helpers import fake_app_manifest


class ArtifactCompatibilityTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(os, "O_PATH"), "retained deletion requires Linux")
    def test_retained_recursive_removal_handles_entry_types_and_fsync_order(
        self,
    ) -> None:
        """Remove only retained entries and flush each directory before its parent."""
        from src import artifact_compatibility

        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir) / "parent"
            target = parent / "target"
            nested = target / "nested"
            nested.mkdir(parents=True)
            outside = Path(tmpdir) / "outside"
            outside.write_bytes(b"outside")
            (target / "file").write_bytes(b"file")
            (nested / "child").write_bytes(b"child")
            (target / "link").symlink_to(outside)
            fifo = target / "fifo"
            if hasattr(os, "mkfifo"):
                os.mkfifo(fifo)

            chain = RetainedDirectoryChain.open(parent)
            with chain:
                descriptor = chain.open_child(
                    "target",
                    os.O_RDONLY | os.O_DIRECTORY,
                )
                target_identity = os.fstat(descriptor).st_ino
                nested_identity = nested.stat().st_ino
                real_fsync = os.fsync
                synced: list[int] = []

                def record_fsync(directory_descriptor: int) -> None:
                    synced.append(os.fstat(directory_descriptor).st_ino)
                    real_fsync(directory_descriptor)

                try:
                    with patch.object(
                        artifact_compatibility.os,
                        "fsync",
                        side_effect=record_fsync,
                    ):
                        chain.remove_child_tree("target", descriptor)
                finally:
                    os.close(descriptor)

            self.assertFalse(target.exists())
            self.assertEqual(outside.read_bytes(), b"outside")
            self.assertIn(target_identity, synced)
            self.assertLess(
                max(
                    index
                    for index, identity in enumerate(synced)
                    if identity == nested_identity
                ),
                max(
                    index
                    for index, identity in enumerate(synced)
                    if identity == target_identity
                ),
            )
            self.assertEqual(synced[-1], parent.stat().st_ino)

    def test_retained_recursive_removal_preserves_replacements_at_boundaries(
        self,
    ) -> None:
        """Fail closed when traversal, descent, or commit sees a replacement."""
        from src import artifact_compatibility

        for boundary in ("traversal", "descent", "commit"):
            with (
                self.subTest(boundary=boundary),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                parent = Path(tmpdir) / "parent"
                target = parent / "target"
                nested = target / "nested"
                nested.mkdir(parents=True)
                (nested / "file").write_bytes(b"owned")
                parked = parent / "parked"
                chain = RetainedDirectoryChain.open(parent)
                with chain:
                    descriptor = chain.open_child(
                        "target",
                        os.O_RDONLY | os.O_DIRECTORY,
                    )
                    real_listdir = os.listdir
                    real_open = os.open
                    real_rename = os.rename
                    replaced = False

                    def replace_target() -> None:
                        nonlocal replaced
                        if replaced:
                            return
                        replaced = True
                        if target.exists():
                            real_rename(target, parked)
                        target.mkdir()
                        (target / "replacement").write_bytes(b"replacement")

                    def replace_during_listdir(value):
                        if boundary == "traversal" and value == descriptor:
                            replace_target()
                        return real_listdir(value)

                    def replace_during_open(name, flags, *args, **kwargs):
                        if boundary == "descent" and name == "nested":
                            replace_target()
                        return real_open(name, flags, *args, **kwargs)

                    def replace_during_rename(source, destination, *args, **kwargs):
                        if boundary == "commit" and source == "target":
                            replace_target()
                        return real_rename(source, destination, *args, **kwargs)

                    patches = (
                        patch.object(
                            artifact_compatibility.os,
                            "listdir",
                            side_effect=replace_during_listdir,
                        ),
                        patch.object(
                            artifact_compatibility.os,
                            "open",
                            side_effect=replace_during_open,
                        ),
                        patch.object(
                            artifact_compatibility.os,
                            "rename",
                            side_effect=replace_during_rename,
                        ),
                    )
                    try:
                        with patches[0], patches[1], patches[2]:
                            if boundary == "commit":
                                with self.assertRaises((OSError, ValueError)):
                                    chain.remove_child_tree("target", descriptor)
                            else:
                                chain.remove_child_tree("target", descriptor)
                    finally:
                        os.close(descriptor)

                self.assertEqual(
                    (target / "replacement").read_bytes(),
                    b"replacement",
                )

    def test_retained_recursive_removal_preserves_concurrent_replacement(
        self,
    ) -> None:
        """Delete the detached retained inode while a concurrent public replacement wins."""
        from src import artifact_compatibility

        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir) / "parent"
            target = parent / "target"
            target.mkdir(parents=True)
            (target / "file").write_bytes(b"owned")
            chain = RetainedDirectoryChain.open(parent)
            with chain:
                descriptor = chain.open_child(
                    "target",
                    os.O_RDONLY | os.O_DIRECTORY,
                )
                real_listdir = os.listdir
                traversing = threading.Event()
                release = threading.Event()
                paused = False

                def pause_traversal(value):
                    nonlocal paused
                    if value == descriptor and not paused:
                        paused = True
                        traversing.set()
                        self.assertTrue(release.wait(timeout=2))
                    return real_listdir(value)

                def install_replacement() -> None:
                    self.assertTrue(traversing.wait(timeout=2))
                    target.mkdir()
                    (target / "replacement").write_bytes(b"replacement")
                    release.set()

                thread = threading.Thread(target=install_replacement)
                thread.start()
                try:
                    with patch.object(
                        artifact_compatibility.os,
                        "listdir",
                        side_effect=pause_traversal,
                    ):
                        chain.remove_child_tree("target", descriptor)
                finally:
                    os.close(descriptor)
                    release.set()
                    thread.join(timeout=2)

            self.assertFalse(thread.is_alive())
            self.assertEqual(
                (target / "replacement").read_bytes(),
                b"replacement",
            )

    def test_retained_recursive_removal_keeps_partial_failure_private_for_retry(
        self,
    ) -> None:
        """Keep a partial tree private and finish it without restoring its name."""
        from src import artifact_compatibility

        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir) / "parent"
            target = parent / "target"
            target.mkdir(parents=True)
            (target / "file").write_bytes(b"owned")
            chain = RetainedDirectoryChain.open(parent)
            with chain:
                descriptor = chain.open_child(
                    "target",
                    os.O_RDONLY | os.O_DIRECTORY,
                )
                real_unlink = os.unlink
                failed = False

                def fail_first_unlink(name, *args, **kwargs):
                    nonlocal failed
                    if not failed and str(name).endswith(".deleting"):
                        failed = True
                        raise OSError("injected unlink failure")
                    return real_unlink(name, *args, **kwargs)

                try:
                    with (
                        patch.object(
                            artifact_compatibility.os,
                            "unlink",
                            side_effect=fail_first_unlink,
                        ),
                        self.assertRaisesRegex(OSError, "unlink failure"),
                    ):
                        chain.remove_child_tree("target", descriptor)
                    self.assertFalse(target.exists())
                    tombstones = list(parent.glob("*.deleting"))
                    self.assertEqual(len(tombstones), 1)
                    self.assertEqual((tombstones[0] / "file").read_bytes(), b"owned")
                    chain.remove_detached_child_tree(tombstones[0].name, descriptor)
                finally:
                    os.close(descriptor)

            self.assertFalse(target.exists())
            self.assertFalse(any(parent.glob("*.deleting")))

    def test_retained_chain_creation_preserves_unproved_suffixes_and_retries(
        self,
    ) -> None:
        """Preserve post-mkdir names when initial identity capture fails."""
        from src import artifact_compatibility

        for replacement in (False, True):
            with (
                self.subTest(replacement=replacement),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                target = root / "first" / "second" / "third" / "fourth"
                parked = root / "first" / "second" / "parked"
                real_mkdir = os.mkdir
                real_rename = os.rename
                real_stat = os.stat
                baseline = len(os.listdir("/proc/self/fd"))
                created_third = False
                failed = False
                replacement_identity: int | None = None

                def record_mkdir(name, mode=0o777, *, dir_fd=None):
                    nonlocal created_third
                    result = real_mkdir(name, mode=mode, dir_fd=dir_fd)
                    if name == "third":
                        created_third = True
                    return result

                def fail_identity_capture(name, *args, **kwargs):
                    nonlocal failed, replacement_identity
                    result = real_stat(name, *args, **kwargs)
                    if (
                        created_third
                        and not failed
                        and name == "third"
                        and kwargs.get("dir_fd") is not None
                    ):
                        failed = True
                        if replacement:
                            parent_descriptor = kwargs["dir_fd"]
                            real_rename(
                                "third",
                                "parked",
                                src_dir_fd=parent_descriptor,
                                dst_dir_fd=parent_descriptor,
                            )
                            real_mkdir("third", dir_fd=parent_descriptor)
                            replacement_identity = real_stat(
                                "third",
                                dir_fd=parent_descriptor,
                                follow_symlinks=False,
                            ).st_ino
                        raise OSError("injected identity capture failure")
                    return result

                with (
                    patch.object(
                        artifact_compatibility.os,
                        "mkdir",
                        side_effect=record_mkdir,
                    ),
                    patch.object(
                        artifact_compatibility.os,
                        "stat",
                        side_effect=fail_identity_capture,
                    ),
                    self.assertRaisesRegex(OSError, "identity capture failure"),
                ):
                    RetainedDirectoryChain.open(
                        target,
                        create=True,
                        check_platform=False,
                    )

                self.assertTrue(target.parent.is_dir())
                self.assertEqual(len(os.listdir("/proc/self/fd")), baseline)
                if replacement:
                    self.assertTrue(parked.is_dir())
                    self.assertEqual(target.parent.stat().st_ino, replacement_identity)

                with RetainedDirectoryChain.open(
                    target,
                    create=True,
                    check_platform=False,
                ) as chain:
                    chain.commit()

                self.assertTrue(target.is_dir())
                if replacement:
                    self.assertEqual(target.parent.stat().st_ino, replacement_identity)

    def test_retained_chain_creation_cleans_proven_failures_and_retries(
        self,
    ) -> None:
        """Clean proven empty suffixes after each creation-stage failure without leaks."""
        from src import artifact_compatibility

        for operation in ("mkdir", "open", "validation", "fsync"):
            with (
                self.subTest(operation=operation),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                target = Path(tmpdir) / "first" / "second" / "third"
                real_mkdir = os.mkdir
                real_open = os.open
                real_fstat = os.fstat
                real_fsync = os.fsync
                baseline = len(os.listdir("/proc/self/fd"))
                failed = False
                created_third = False

                def fail_mkdir(name, mode=0o777, *, dir_fd=None):
                    nonlocal created_third
                    if operation == "mkdir" and name == "third":
                        raise OSError("injected mkdir failure")
                    result = real_mkdir(name, mode=mode, dir_fd=dir_fd)
                    if name == "third":
                        created_third = True
                    return result

                def fail_open(name, flags, *args, **kwargs):
                    nonlocal failed
                    if (
                        operation == "open"
                        and not failed
                        and name == "third"
                        and kwargs.get("dir_fd") is not None
                    ):
                        failed = True
                        raise OSError("injected open failure")
                    return real_open(name, flags, *args, **kwargs)

                def fail_fstat(descriptor):
                    nonlocal failed
                    if operation == "validation" and not failed:
                        try:
                            selected = os.readlink(f"/proc/self/fd/{descriptor}")
                        except OSError:
                            selected = ""
                        if selected.endswith("/third"):
                            failed = True
                            raise OSError("injected validation failure")
                    return real_fstat(descriptor)

                def fail_fsync(descriptor):
                    nonlocal failed
                    if operation == "fsync" and created_third and not failed:
                        failed = True
                        raise OSError("injected fsync failure")
                    return real_fsync(descriptor)

                with (
                    patch.object(
                        artifact_compatibility.os,
                        "mkdir",
                        side_effect=fail_mkdir,
                    ),
                    patch.object(
                        artifact_compatibility.os,
                        "open",
                        side_effect=fail_open,
                    ),
                    patch.object(
                        artifact_compatibility.os,
                        "fstat",
                        side_effect=fail_fstat,
                    ),
                    patch.object(
                        artifact_compatibility.os,
                        "fsync",
                        side_effect=fail_fsync,
                    ),
                    self.assertRaisesRegex(OSError, f"{operation} failure"),
                ):
                    RetainedDirectoryChain.open(
                        target,
                        create=True,
                        check_platform=False,
                    )

                self.assertFalse((Path(tmpdir) / "first").exists())
                self.assertEqual(len(os.listdir("/proc/self/fd")), baseline)
                with RetainedDirectoryChain.open(
                    target,
                    create=True,
                    check_platform=False,
                ) as chain:
                    chain.commit()
                self.assertTrue(target.is_dir())

    def test_retained_chain_creation_preserves_open_boundary_replacement(
        self,
    ) -> None:
        """Keep a replacement installed after identity capture and allow retry."""
        from src import artifact_compatibility

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "first" / "second" / "third"
            replacement = Path(tmpdir) / "replacement"
            real_open = os.open
            failed = False

            def replace_before_open(name, flags, *args, **kwargs):
                nonlocal failed
                if not failed and name == "second" and kwargs.get("dir_fd") is not None:
                    failed = True
                    second = Path(tmpdir) / "first" / "second"
                    second.rename(replacement)
                    second.mkdir()
                    (second / "replacement").write_bytes(b"replacement")
                    raise OSError("injected replacement open failure")
                return real_open(name, flags, *args, **kwargs)

            with (
                patch.object(
                    artifact_compatibility.os,
                    "open",
                    side_effect=replace_before_open,
                ),
                self.assertRaisesRegex(OSError, "replacement open failure"),
            ):
                RetainedDirectoryChain.open(
                    target,
                    create=True,
                    check_platform=False,
                )

            marker = Path(tmpdir) / "first" / "second" / "replacement"
            self.assertEqual(marker.read_bytes(), b"replacement")
            with RetainedDirectoryChain.open(
                target,
                create=True,
                check_platform=False,
            ) as chain:
                chain.commit()
            self.assertEqual(marker.read_bytes(), b"replacement")
            self.assertTrue(target.is_dir())

    def test_strict_json_rejects_overflow_and_accepts_finite_exponents(self) -> None:
        for value in ("1e999", "-1e999"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "invalid test document"),
            ):
                strict_json_loads(f'{{"value":{value}}}', source="test document")

        self.assertEqual(
            strict_json_loads('{"upper":1e308,"lower":-1e308}', source="test document"),
            {"upper": 1e308, "lower": -1e308},
        )

    def test_secure_artifact_platform_rejects_windows(self) -> None:
        with (
            patch("src.artifact_compatibility.sys.platform", "win32"),
            self.assertRaisesRegex(RuntimeError, "require Linux"),
        ):
            require_secure_artifact_platform()

    def test_reader_has_no_unsafe_fallback_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "artifact.json"
            artifact.write_text("{}", encoding="utf-8")
            with (
                patch("src.artifact_compatibility.sys.platform", "win32"),
                patch("src.artifact_compatibility.os.open") as open_file,
                self.assertRaisesRegex(RuntimeError, "require Linux"),
            ):
                read_regular_file(artifact, parent=root)

            open_file.assert_not_called()

    @unittest.skipUnless(hasattr(os, "O_TMPFILE"), "immutable writes require Linux")
    def test_immutable_atomic_write_has_one_name_and_preserves_collision(
        self,
    ) -> None:
        """Publish an unnamed inode once and leave an existing destination exact."""
        from src import artifact_compatibility

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "artifact.bin"
            real_link = artifact_compatibility.link_unnamed_file_at
            observed: list[tuple[set[str], int]] = []

            def inspect_publication(
                source_descriptor: int,
                destination_descriptor: int,
                destination_name: str,
            ) -> None:
                """Record the complete namespace immediately after publication.

                Parameters
                ----------
                source_descriptor : int
                    Open unnamed source descriptor.
                destination_descriptor : int
                    Retained destination-directory descriptor.
                destination_name : str
                    Direct destination child name.

                Returns
                -------
                None
                """
                real_link(
                    source_descriptor,
                    destination_descriptor,
                    destination_name,
                )
                installed = os.stat(
                    destination_name,
                    dir_fd=destination_descriptor,
                    follow_symlinks=False,
                )
                observed.append(
                    (set(os.listdir(destination_descriptor)), installed.st_nlink)
                )

            with patch.object(
                artifact_compatibility,
                "link_unnamed_file_at",
                side_effect=inspect_publication,
            ):
                self.assertEqual(
                    write_bytes_atomically(target, b"published", overwrite=False),
                    target,
                )

            self.assertEqual(observed, [({target.name}, 1)])
            original = target.stat(follow_symlinks=False)
            baseline_descriptors = len(os.listdir("/proc/self/fd"))
            with self.assertRaises(FileExistsError) as raised:
                write_bytes_atomically(target, b"replacement", overwrite=False)

            current = target.stat(follow_symlinks=False)
            self.assertIs(type(raised.exception), FileExistsError)
            self.assertEqual(target.read_bytes(), b"published")
            self.assertEqual(
                (current.st_dev, current.st_ino, current.st_mode, current.st_nlink),
                (original.st_dev, original.st_ino, original.st_mode, 1),
            )
            self.assertEqual(set(root.iterdir()), {target})
            self.assertEqual(len(os.listdir("/proc/self/fd")), baseline_descriptors)

    @unittest.skipUnless(hasattr(os, "O_TMPFILE"), "immutable writes require Linux")
    def test_immutable_atomic_write_cleans_failed_source_and_rejects_linked_parent(
        self,
    ) -> None:
        """Close unnamed sources on failure and never traverse a linked parent."""
        from src import artifact_compatibility

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_descriptors = len(os.listdir("/proc/self/fd"))
            target = root / "artifact.bin"
            with (
                patch.object(
                    artifact_compatibility,
                    "link_unnamed_file_at",
                    side_effect=RuntimeError("linkat unavailable"),
                ),
                self.assertRaisesRegex(RuntimeError, "linkat unavailable"),
            ):
                write_bytes_atomically(target, b"content", overwrite=False)

            self.assertEqual(list(root.iterdir()), [])
            self.assertEqual(len(os.listdir("/proc/self/fd")), baseline_descriptors)

            real_open = artifact_compatibility.os.open

            def reject_unnamed_file(name, flags, *args, **kwargs):
                """Inject an unavailable ``O_TMPFILE`` primitive.

                Parameters
                ----------
                name : object
                    Name passed to ``os.open``.
                flags : int
                    Flags passed to ``os.open``.
                *args : object
                    Positional arguments passed to ``os.open``.
                **kwargs : object
                    Keyword arguments passed to ``os.open``.

                Returns
                -------
                int
                    Descriptor returned by the real ``os.open``.
                """
                if name == "." and flags & os.O_TMPFILE:
                    raise OSError(
                        errno.EOPNOTSUPP,
                        "injected O_TMPFILE failure",
                    )
                return real_open(name, flags, *args, **kwargs)

            with (
                patch.object(
                    artifact_compatibility,
                    "require_secure_artifact_platform",
                ),
                patch.object(
                    artifact_compatibility.os,
                    "open",
                    side_effect=reject_unnamed_file,
                ),
                self.assertRaisesRegex(RuntimeError, "O_TMPFILE support is required"),
            ):
                write_bytes_atomically(target, b"content", overwrite=False)

            self.assertEqual(list(root.iterdir()), [])
            self.assertEqual(len(os.listdir("/proc/self/fd")), baseline_descriptors)

            outside = root / "outside"
            outside.mkdir()
            linked_parent = root / "linked"
            linked_parent.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "artifact directory chain changed"):
                write_bytes_atomically(
                    linked_parent / target.name,
                    b"content",
                    overwrite=False,
                )

            self.assertEqual(list(outside.iterdir()), [])
            self.assertEqual(len(os.listdir("/proc/self/fd")), baseline_descriptors)

    @unittest.skipUnless(hasattr(os, "O_TMPFILE"), "immutable writes require Linux")
    def test_immutable_atomic_write_rejects_replacement_at_every_boundary(
        self,
    ) -> None:
        """Reject foreign names after link, fsync, and both return boundaries."""
        from src import artifact_compatibility

        real_verify = artifact_compatibility.verify_published_unnamed_file_at
        for boundary in range(1, 5):
            with (
                self.subTest(boundary=boundary),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                target = root / "artifact.bin"
                calls = 0

                def replace_before_verify(
                    source_descriptor: int,
                    snapshot,
                    parent_descriptor: int,
                    name: str,
                    *,
                    expected_content: bytes,
                ) -> None:
                    """Replace the published name before one selected verification.

                    Parameters
                    ----------
                    source_descriptor : int
                        Retained unnamed source descriptor.
                    snapshot : RegularFileSnapshot
                        Captured source snapshot.
                    parent_descriptor : int
                        Retained destination-directory descriptor.
                    name : str
                        Direct destination child name.
                    expected_content : bytes
                        Exact expected destination bytes.

                    Returns
                    -------
                    None
                    """
                    nonlocal calls
                    calls += 1
                    if calls == boundary:
                        os.unlink(name, dir_fd=parent_descriptor)
                        replacement = os.open(
                            name,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                            0o600,
                            dir_fd=parent_descriptor,
                        )
                        try:
                            os.write(replacement, b"foreign")
                        finally:
                            os.close(replacement)
                    real_verify(
                        source_descriptor,
                        snapshot,
                        parent_descriptor,
                        name,
                        expected_content=expected_content,
                    )

                baseline_descriptors = len(os.listdir("/proc/self/fd"))
                with (
                    patch.object(
                        artifact_compatibility,
                        "verify_published_unnamed_file_at",
                        side_effect=replace_before_verify,
                    ),
                    self.assertRaisesRegex(ValueError, "changed while retained"),
                ):
                    write_bytes_atomically(target, b"published", overwrite=False)

                self.assertEqual(target.read_bytes(), b"foreign")
                self.assertEqual(set(root.iterdir()), {target})
                self.assertEqual(len(os.listdir("/proc/self/fd")), baseline_descriptors)

    @unittest.skipUnless(hasattr(os, "O_TMPFILE"), "immutable writes require Linux")
    def test_immutable_atomic_write_classifies_only_capability_open_errors(
        self,
    ) -> None:
        """Preserve operational ``O_TMPFILE`` failures and classify support errors."""
        from src import artifact_compatibility

        unsupported = (errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL, errno.EISDIR)
        operational = (
            errno.ENOSPC,
            errno.EDQUOT,
            errno.EACCES,
            errno.EROFS,
            errno.EINTR,
            errno.EMFILE,
            errno.ENFILE,
        )
        real_open = artifact_compatibility.os.open
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_descriptors = len(os.listdir("/proc/self/fd"))
            for error_number in (*unsupported, *operational):
                with self.subTest(error_number=error_number):
                    injected = OSError(error_number, os.strerror(error_number))

                    def reject_unnamed_file(name, flags, *args, **kwargs):
                        """Inject one selected ``O_TMPFILE`` open error.

                        Parameters
                        ----------
                        name : object
                            Name passed to ``os.open``.
                        flags : int
                            Flags passed to ``os.open``.
                        *args : object
                            Positional arguments passed to ``os.open``.
                        **kwargs : object
                            Keyword arguments passed to ``os.open``.

                        Returns
                        -------
                        int
                            Descriptor returned by the real ``os.open``.
                        """
                        if name == "." and flags & os.O_TMPFILE:
                            raise injected
                        return real_open(name, flags, *args, **kwargs)

                    context = (
                        self.assertRaises(RuntimeError)
                        if error_number in unsupported
                        else self.assertRaises(OSError)
                    )
                    with (
                        patch.object(
                            artifact_compatibility,
                            "require_secure_artifact_platform",
                        ),
                        patch.object(
                            artifact_compatibility.os,
                            "open",
                            side_effect=reject_unnamed_file,
                        ),
                        context as raised,
                    ):
                        write_bytes_atomically(
                            root / f"artifact-{error_number}.bin",
                            b"content",
                            overwrite=False,
                        )

                    if error_number in operational:
                        self.assertIs(raised.exception, injected)
                        self.assertEqual(raised.exception.errno, error_number)
                    self.assertEqual(list(root.iterdir()), [])
                    self.assertEqual(
                        len(os.listdir("/proc/self/fd")), baseline_descriptors
                    )

    def test_compatibility_policy_records_client_schema_two_migration(self) -> None:
        policy = (
            Path(__file__).resolve().parent.parent / "COMPATIBILITY.md"
        ).read_text(encoding="utf-8")

        self.assertIn("client-N/client_metadata.json` uses `schema_version: 2`", policy)
        self.assertIn("client schema `1` shards", policy)
        self.assertIn("Regenerate public schema\n`2`, client schema `2`", policy)
        self.assertIn("schema-3 generation atomically supersedes it", policy)
        self.assertIn("server artifact manifests use schema `4`", policy)
        self.assertIn(
            "Server schema `3` did not retain canonical public evidence", policy
        )
        self.assertNotIn("schema-2 generation atomically supersedes it", policy)
        self.assertNotIn("all other persisted contracts remain schema `1`", policy)

    def test_current_schema_is_supported(self) -> None:
        payload = {"schema_version": ARTIFACT_SCHEMA_VERSION}

        self.assertIs(validate_artifact_schema(payload, "test artifact"), payload)

    def test_unversioned_schema_is_rejected_with_regeneration_guidance(self) -> None:
        with self.assertRaisesRegex(ValueError, "regenerate"):
            validate_artifact_schema({}, "test artifact")

    def test_older_schema_is_rejected_with_regeneration_guidance(self) -> None:
        with self.assertRaisesRegex(ValueError, "older.*regenerate"):
            validate_artifact_schema({"schema_version": 0}, "test artifact")

    def test_newer_schema_is_rejected_with_regeneration_guidance(self) -> None:
        with self.assertRaisesRegex(ValueError, "newer.*regenerate"):
            validate_artifact_schema({"schema_version": 2}, "test artifact")

    def test_public_manifest_rejects_incompatible_schema_before_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "manifest.json").write_text(
                json.dumps({"schema_version": PUBLIC_ARTIFACT_SCHEMA_VERSION + 1}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "public manifest.*newer"):
                load_app_manifest(public_artifact_dir=path)

    def test_public_manifest_checks_each_schema_version_state(self) -> None:
        vocabulary = b"\n[UNK]\ngood\n"
        vocabulary_sha256 = hashlib.sha256(vocabulary).hexdigest()
        dataset = {
            "id": "example/imdb",
            "config": "plain_text",
            "revision": "frozen",
            "datasets_version": "1.0.0",
            "split": "train",
            "rows": 3,
            "raw_parquet_sha256": "1" * 64,
            "content_sha256": "2" * 64,
        }
        protocol = {
            "dataset": {
                **{
                    key: dataset[key]
                    for key in (
                        "id",
                        "config",
                        "revision",
                        "datasets_version",
                    )
                },
                "splits": {
                    "train": {
                        key: dataset[key]
                        for key in (
                            "rows",
                            "raw_parquet_sha256",
                            "content_sha256",
                        )
                    }
                },
            },
            "preprocessing": {
                "vocabulary_size": 3,
                "max_tokens": 3,
                "output_sequence_length": 500,
                "vocabulary_sha256": vocabulary_sha256,
            },
            "model": {
                "vocabulary_size": 3,
                "sequence_length": 500,
                "embedding_dimension": 100,
            },
        }
        valid_payload = {
            "schema_version": PUBLIC_ARTIFACT_SCHEMA_VERSION,
            "embedding_dim": 100,
            "sequence_length": 500,
            "vocabulary_size": 3,
            "vocabulary": {
                "filename": "vocab.txt",
                "sha256": vocabulary_sha256,
                "size_bytes": len(vocabulary),
            },
            "dataset": dataset,
        }
        for version, error in (
            (None, "no valid"),
            (1, "older.*regenerate or migrate"),
            (PUBLIC_ARTIFACT_SCHEMA_VERSION + 1, "newer"),
        ):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)
                payload = dict(valid_payload)
                if version is None:
                    payload.pop("schema_version")
                else:
                    payload["schema_version"] = version
                (path / "manifest.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

                with self.assertRaisesRegex(ValueError, error):
                    load_app_manifest(public_artifact_dir=path)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "vocab.txt").write_bytes(vocabulary)
            (path / "manifest.json").write_bytes(canonical_json_bytes(valid_payload))
            manifest = load_app_manifest(public_artifact_dir=path, protocol=protocol)
            self.assertEqual(manifest.payload, valid_payload)
            with self.assertRaises(TypeError):
                manifest.payload["dataset"]["id"] = "mutated"
            with self.assertRaises(TypeError):
                manifest.payload["vocabulary"]["sha256"] = "0" * 64

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "vocab.txt").write_bytes(vocabulary)
            nested_payload = {
                **valid_payload,
                "producer": {"history": [{"revision": "frozen"}]},
            }
            (path / "manifest.json").write_bytes(canonical_json_bytes(nested_payload))
            manifest = load_app_manifest(public_artifact_dir=path, protocol=protocol)
            with self.assertRaises(TypeError):
                manifest.payload["producer"]["history"][0]["revision"] = "mutated"

        invalid_dimensions = (
            ("embedding_dim", -3),
            ("sequence_length", 4.75),
            ("sequence_length", True),
            ("vocabulary_size", 4),
        )
        for field, value in invalid_dimensions:
            with (
                self.subTest(field=field, value=value),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                path = Path(tmpdir)
                (path / "vocab.txt").write_bytes(vocabulary)
                payload = {**valid_payload, field: value}
                (path / "manifest.json").write_bytes(canonical_json_bytes(payload))
                with self.assertRaisesRegex(ValueError, "protocol dimensions"):
                    load_app_manifest(public_artifact_dir=path, protocol=protocol)

    def test_server_artifact_manifest_declares_supported_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            manifest_path = write_server_artifact_manifest(
                path, app_manifest=fake_app_manifest()
            )

            self.assertEqual(
                canonical_json_bytes(
                    {"artifacts": load_server_artifact_manifest(path)["artifacts"]}
                ),
                canonical_json_bytes({"artifacts": SERVER_ARTIFACTS}),
            )
            self.assertEqual(manifest_path.parent, path)

    def test_server_artifact_manifest_accepts_additive_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            manifest_path = write_server_artifact_manifest(
                path, app_manifest=fake_app_manifest()
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["producer"] = {"upper": 1e308, "lower": -1e308}
            payload["artifacts"]["model"]["description"] = "global model"
            payload["artifacts"]["diagnostics"] = {"filename": "debug.json"}
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertEqual(
                canonical_json_bytes(load_server_artifact_manifest(path)),
                canonical_json_bytes(payload),
            )

    def test_in_progress_snapshot_ignores_unmanifested_writer_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "global_model.keras").write_bytes(b"model-in-progress")
            (path / ".metrics.csv.tmp").write_bytes(b"partial")
            (path / "writer-state").mkdir()
            write_server_artifact_manifest(path, app_manifest=fake_app_manifest())

            snapshot = load_server_artifact_snapshot(path)

            self.assertEqual(snapshot.files["global_model.keras"], b"model-in-progress")
            self.assertNotIn(".metrics.csv.tmp", snapshot.files)

    def test_server_manifest_read_boundaries_reject_hostile_json(self) -> None:
        for operation in ("load", "finalize"):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)
                manifest_path = write_server_artifact_manifest(
                    path, app_manifest=fake_app_manifest()
                )
                valid = manifest_path.read_text(encoding="utf-8")
            hostile_documents = (
                valid.replace(
                    f'"schema_version": {SERVER_ARTIFACT_SCHEMA_VERSION},',
                    '"schema_version": 1,\n  '
                    f'"schema_version": {SERVER_ARTIFACT_SCHEMA_VERSION},',
                    1,
                ),
                valid.replace(
                    '"public_manifest_checksum":',
                    '"public_manifest_checksum": "invalid",\n'
                    '    "public_manifest_checksum":',
                    1,
                ),
                valid.replace(
                    '"vocabulary_size": 20000,',
                    '"vocabulary_size": 1,\n      "vocabulary_size": 20000,',
                    1,
                ),
                *(
                    valid[:-2] + f',\n  "producer": {constant}\n}}\n'
                    for constant in (
                        "NaN",
                        "Infinity",
                        "-Infinity",
                        "1e999",
                        "-1e999",
                    )
                ),
            )
            for document in hostile_documents:
                with (
                    self.subTest(operation=operation, document=document),
                    tempfile.TemporaryDirectory() as tmpdir,
                ):
                    path = Path(tmpdir)
                    manifest_path = write_server_artifact_manifest(
                        path, app_manifest=fake_app_manifest()
                    )
                    manifest_path.write_text(document, encoding="utf-8")

                    with self.assertRaisesRegex(
                        ValueError, "invalid (existing )?server artifact manifest"
                    ):
                        if operation == "load":
                            load_server_artifact_snapshot(path)
                        else:
                            write_server_artifact_manifest(
                                path,
                                app_manifest=fake_app_manifest(),
                                finalized=True,
                            )

    def test_server_artifact_manifest_rejects_changed_layout_without_version(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            write_server_artifact_manifest(path, app_manifest=fake_app_manifest())
            manifest_path = path / "artifact_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["artifacts"]["metrics"]["columns"].append("unsupported")
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not match.*regenerate"):
                load_server_artifact_manifest(path)

    def test_server_snapshot_binding_is_exact_and_deeply_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            app_manifest = fake_app_manifest()
            manifest_path = write_server_artifact_manifest(
                path, app_manifest=app_manifest
            )
            snapshot = load_server_artifact_snapshot(path, app_manifest=app_manifest)

            with self.assertRaises(TypeError):
                snapshot.manifest["binding"]["model_dimensions"]["sequence_length"] = 1
            with self.assertRaises(TypeError):
                snapshot.manifest["artifacts"]["metrics"]["columns"][0] = "epoch"

            for field, value in (
                ("public_manifest_checksum", "sha256:" + "0" * 64),
                ("vocabulary_checksum", "sha256:" + "1" * 64),
                ("model_dimensions", {"vocabulary_size": 1}),
            ):
                with self.subTest(field=field):
                    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                    payload["binding"][field] = value
                    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "bound|binding"):
                        load_server_artifact_snapshot(path, app_manifest=app_manifest)
                    write_server_artifact_manifest(path, app_manifest=app_manifest)

    def test_schema_one_server_artifact_requires_regeneration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            manifest_path = write_server_artifact_manifest(
                path, app_manifest=fake_app_manifest()
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["schema_version"] = 1
            payload.pop("binding")
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "older.*regenerate"):
                load_server_artifact_snapshot(path)

    def test_unversioned_server_artifact_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "schema_version.*regenerate"):
                load_server_artifact_manifest(Path(tmpdir))

    def test_server_artifact_manifest_rejects_older_and_newer_schemas(self) -> None:
        for version, direction in (
            (SERVER_ARTIFACT_SCHEMA_VERSION - 1, "older"),
            (SERVER_ARTIFACT_SCHEMA_VERSION + 1, "newer"),
        ):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)
                manifest_path = write_server_artifact_manifest(
                    path, app_manifest=fake_app_manifest()
                )
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                payload["schema_version"] = version
                manifest_path.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, direction):
                    load_server_artifact_manifest(path)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO files require POSIX")
    def test_finalization_rejects_non_regular_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            for name in ("global_model.keras", "metrics.csv", "run_manifest.json"):
                (path / name).touch()
            os.mkfifo(path / "unexpected.pipe")

            with self.assertRaisesRegex(ValueError, "contained regular file"):
                write_server_artifact_manifest(
                    path, app_manifest=fake_app_manifest(), finalized=True
                )

    def test_finalization_rejects_hard_linked_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run"
            path.mkdir()
            outside = Path(tmpdir) / "outside.keras"
            outside.touch()
            os.link(outside, path / "global_model.keras")
            for name in ("metrics.csv", "run_manifest.json"):
                (path / name).touch()

            with self.assertRaisesRegex(ValueError, "contained regular file"):
                write_server_artifact_manifest(
                    path, app_manifest=fake_app_manifest(), finalized=True
                )


if __name__ == "__main__":
    unittest.main()
