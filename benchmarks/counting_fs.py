"""FUSE passthrough filesystem that counts read I/O.

Requires ``pyfuse3`` and ``trio``.  The benchmark runner gracefully
falls back to speed-only measurement when these are unavailable.
"""

from __future__ import annotations

import errno
import os
import threading

import pyfuse3


class CountingFS(pyfuse3.Operations):
    """Passthrough FS that transparently counts bytes read and read calls."""

    def __init__(self, root: str):
        super().__init__()
        self.root = root
        self.read_bytes = 0
        self.read_calls = 0
        self._lock = threading.Lock()
        self._fd_map: dict[int, int] = {}
        self._inode_path: dict[int, str] = {pyfuse3.ROOT_INODE: root}
        self._path_inode: dict[str, int] = {root: pyfuse3.ROOT_INODE}
        self._next_inode = pyfuse3.ROOT_INODE + 1

    # ---- helpers -----------------------------------------------------------
    def _get_path(self, inode: int) -> str | None:
        return self._inode_path.get(inode)

    def _get_or_create_inode(self, path: str) -> int:
        if path in self._path_inode:
            return self._path_inode[path]
        inode = self._next_inode
        self._next_inode += 1
        self._inode_path[inode] = path
        self._path_inode[path] = inode
        return inode

    def reset_stats(self) -> None:
        with self._lock:
            self.read_bytes = 0
            self.read_calls = 0

    def get_stats(self) -> dict[str, int]:
        with self._lock:
            return {"bytes": self.read_bytes, "calls": self.read_calls}

    # ---- FUSE ops ----------------------------------------------------------
    async def getattr(self, inode, ctx=None):
        path = self._get_path(inode)
        if path is None:
            raise pyfuse3.FUSEError(errno.ENOENT)
        try:
            st = os.lstat(path)
        except OSError as exc:
            raise pyfuse3.FUSEError(exc.errno)
        entry = pyfuse3.EntryAttributes()
        entry.st_ino = inode
        entry.st_mode = st.st_mode
        entry.st_nlink = st.st_nlink
        entry.st_uid = st.st_uid
        entry.st_gid = st.st_gid
        entry.st_size = st.st_size
        entry.st_atime_ns = int(st.st_atime * 1e9)
        entry.st_mtime_ns = int(st.st_mtime * 1e9)
        entry.st_ctime_ns = int(st.st_ctime * 1e9)
        entry.st_blksize = 512
        entry.st_blocks = (st.st_size + 511) // 512
        return entry

    async def lookup(self, parent_inode, name, ctx=None):
        parent_path = self._get_path(parent_inode)
        if parent_path is None:
            raise pyfuse3.FUSEError(errno.ENOENT)
        name = name.decode("utf-8") if isinstance(name, bytes) else name
        path = os.path.join(parent_path, name)
        if not os.path.exists(path):
            raise pyfuse3.FUSEError(errno.ENOENT)
        inode = self._get_or_create_inode(path)
        return await self.getattr(inode)

    async def opendir(self, inode, ctx):
        path = self._get_path(inode)
        if path is None or not os.path.isdir(path):
            raise pyfuse3.FUSEError(errno.ENOENT)
        return inode

    async def readdir(self, inode, start_id, token):
        path = self._get_path(inode)
        if path is None:
            raise pyfuse3.FUSEError(errno.ENOENT)
        entries = list(os.listdir(path))
        for i, name in enumerate(entries[start_id:], start=start_id):
            child_path = os.path.join(path, name)
            child_inode = self._get_or_create_inode(child_path)
            attr = await self.getattr(child_inode)
            if not pyfuse3.readdir_reply(token, name.encode(), attr, i + 1):
                break

    async def open(self, inode, flags, ctx):
        path = self._get_path(inode)
        if path is None:
            raise pyfuse3.FUSEError(errno.ENOENT)
        fd = os.open(path, flags)
        # Key by fd (not inode) so concurrent opens of the same file
        # each get their own entry and don't overwrite each other.
        self._fd_map[fd] = fd
        return pyfuse3.FileInfo(fh=fd, direct_io=True)

    async def read(self, fh, offset, size):
        fd = self._fd_map.get(fh)
        if fd is None:
            raise pyfuse3.FUSEError(errno.EBADF)
        os.lseek(fd, offset, os.SEEK_SET)
        data = os.read(fd, size)
        with self._lock:
            self.read_bytes += len(data)
            self.read_calls += 1
        return data

    async def release(self, fh):
        fd = self._fd_map.pop(fh, None)
        if fd is not None:
            os.close(fd)
