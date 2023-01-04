#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Copyright (c) 2014 trgk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import os, sys
import platform
import struct
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_int,
    c_void_p,
    c_wchar_p,
    create_string_buffer,
    sizeof,
)
from tempfile import NamedTemporaryFile

# Ported functions

def u2b(s, encoding = "utf-8"):
    if isinstance(s, str):
        try:
            return s.encode(encoding)
        except (UnicodeEncodeError):
            return s.encode("utf-8")
    elif isinstance(s, bytes):
        return s
    else:
        raise NotImplementedError("Invalid type {}".format(type(s)))

def b2u(b, encoding = "utf-8"):
    if isinstance(b, bytes):
        return b.decode(encoding)
    elif isinstance(b, str):
        return b
    else:
        raise NotImplementedError("Invalid type {}".format(type(s)))

def u2utf8(s):
    return u2b(s, "utf-8")

def find_data_file(filename, file):
    if getattr(sys, "frozen", False):
        # The application is frozen
        datadir = os.path.dirname(sys.executable)
    else:
        # The application is not frozen
        # Change this bit to match where you store your data files:
        datadir = os.path.dirname(file)

    return os.path.join(datadir, filename)

# Constants
MPQ_FILE_COMPRESS = 0x00000200
MPQ_FILE_ENCRYPTED = 0x00010000
MPQ_FILE_FIX_KEY = 0x00020000
MPQ_FILE_REPLACEEXISTING = 0x80000000
MPQ_COMP_ZLIB = 0x00000002

libstorm = None

class CreateInfo(Structure):
    _fields_ = [
        ("cbSize", c_int),
        ("dwMpqVersion", c_int),
        ("pvUserData", c_void_p),
        ("cbUserData", c_int),
        ("dwStreamFlags", c_int),
        ("dwFileFlags1", c_int),
        ("dwFileFlags2", c_int),
        ("dwFileFlags3", c_int),
        ("dwAttrFlags", c_int),
        ("dwSectorSize", c_int),
        ("dwRawChunkSize", c_int),
        ("dwMaxFileCount", c_int),
    ]


def InitMpqLibrary():
    global libstorm

    try:
        platformName = platform.system()
        if platformName == "Windows":  # windows
            from ctypes import WinDLL

            if struct.calcsize("P") == 4:  # 32bit
                libstorm = WinDLL(find_data_file("StormLib32.dll", __file__), use_last_error=True)
            else:  # 64bit
                libstorm = WinDLL(find_data_file("StormLib64.dll", __file__), use_last_error=True)

        elif platformName == "Darwin":  # mac
            from ctypes import CDLL

            try:
                libstorm = CDLL("libstorm.dylib", use_last_error=True)
            except OSError as e:
                raise OSError(
                    "You need to install stormlib before using this program."
                    "$ brew install homebrew/games/stormlib",
                )

        elif platformName == "Linux":  # linux
            from ctypes import CDLL

            try:
                libstorm = CDLL(find_data_file("libstorm.so", __file__), use_last_error=True)
            except OSError as e:
                try:
                    libstorm = CDLL("libstorm.so", use_last_error=True)
                except OSError as e:
                    raise OSError(
                        "You need to install StormLib before using this program."
                        "Please check the installation guide here: https://github.com/ladislav-zezula/StormLib/",
                    )

        # for MpqRead
        libstorm.SFileOpenArchive.restype = c_int
        libstorm.SFileCloseArchive.restype = c_int
        libstorm.SFileOpenFileEx.restype = c_int
        libstorm.SFileGetFileSize.restype = c_int
        libstorm.SFileReadFile.restype = c_int
        libstorm.SFileCloseFile.restype = c_int

        libstorm.SFileCloseArchive.argtypes = [c_void_p]
        libstorm.SFileOpenFileEx.argtypes = [c_void_p, c_char_p, c_int, c_void_p]
        libstorm.SFileGetFileSize.argtypes = [c_void_p, c_void_p]
        libstorm.SFileReadFile.argtypes = [c_void_p, c_char_p, c_int, c_void_p, c_int]
        libstorm.SFileCloseFile.argtypes = [c_void_p]

        # for MpqWrite
        libstorm.SFileCompactArchive.restype = c_int
        libstorm.SFileCreateArchive2.restype = c_int
        libstorm.SFileAddFileEx.restype = c_int
        libstorm.SFileGetMaxFileCount.restype = c_int
        libstorm.SFileSetMaxFileCount.restype = c_int

        libstorm.SFileCompactArchive.argtypes = [c_void_p, c_char_p, c_int]

        libstorm.SFileGetMaxFileCount.argtypes = [c_void_p]
        libstorm.SFileSetMaxFileCount.argtypes = [c_void_p, c_int]

        # Linux fix
        if platformName == "Linux":
          libstorm.SFileOpenArchive.argtypes = [c_char_p, c_int, c_int, c_void_p]
          libstorm.SFileCreateArchive2.argtypes = [
              c_char_p,
              POINTER(CreateInfo),
              c_void_p,
          ]
          libstorm.SFileAddFileEx.argtypes = [
              c_void_p,
              c_char_p,
              c_char_p,
              c_int,
              c_int,
              c_int,
          ]
        else:
          libstorm.SFileOpenArchive.argtypes = [c_wchar_p, c_int, c_int, c_void_p]
          libstorm.SFileCreateArchive2.argtypes = [
            c_wchar_p,
            POINTER(CreateInfo),
            c_void_p,
          ]
          libstorm.SFileAddFileEx.argtypes = [
              c_void_p,
              c_wchar_p,
              c_char_p,
              c_int,
              c_int,
              c_int,
          ]

        return True

    except OSError:
        raise OSError(("Loading StormLib failed."))
        


class MPQ:
    def __init__(self):
        self.mpqh = None
        self.libstorm = libstorm

    def __del__(self):
        self.Close()

    def Open(self, fname):
        if self.mpqh is not None:
            raise RuntimeError(("Duplicate opening"))

        if platform.system() == "Linux":
            fname = u2b(fname)

        h = c_void_p()
        ret = self.libstorm.SFileOpenArchive(fname, 0, 0, byref(h))
        if not ret:
            self.mpqh = None
            return False

        self.mpqh = h
        return True

    def Create(self, fname, *, sectorSize=3, fileCount=1024):
        if self.mpqh is not None:
            raise RuntimeError(("Duplicate opening"))

        cinfo = CreateInfo()
        cinfo.cbSize = sizeof(CreateInfo)
        cinfo.dwMpqVersion = 0
        cinfo.dwStreamFlags = 0
        cinfo.dwFileFlags1 = 0
        cinfo.dwFileFlags2 = 0
        cinfo.dwFileFlags3 = 0
        cinfo.dwAttrFlags = 0
        cinfo.dwSectorSize = 2 ** (9 + sectorSize)
        cinfo.dwMaxFileCount = fileCount

        if platform.system() == "Linux":
            fname = u2b(fname)

        h = c_void_p()
        ret = self.libstorm.SFileCreateArchive2(fname, byref(cinfo), byref(h))
        if not ret:
            self.mpqh = None
            return False

        self.mpqh = h
        return True

    def Close(self):
        if self.mpqh is None:
            return None

        self.libstorm.SFileCloseArchive(self.mpqh)
        self.mpqh = None
        return True

    def EnumFiles(self):
        # using listfile.
        lst = self.Extract("(listfile)")
        if lst is None:
            return []

        try:
            return b2u(lst).replace("\r", "").split("\n")
        except UnicodeDecodeError:
            return []

    # Extract
    def Extract(self, fname):
        if self.libstorm is None:
            return None
        elif not self.mpqh:
            return None
        elif not fname:
            return None

        # Open file
        fileh = c_void_p()
        ret = self.libstorm.SFileOpenFileEx(self.mpqh, u2b(fname), 0, byref(fileh))
        if not ret:
            ret = self.libstorm.SFileOpenFileEx(self.mpqh, u2utf8(fname), 0, byref(fileh))
            if not ret:
                return None

        # Get file size & allocate buffer
        # Note : this version only supports 32bit mpq file
        fsize = self.libstorm.SFileGetFileSize(fileh, 0)
        if not fsize or fsize <= 0:
            return None
        fdata = create_string_buffer(fsize)

        # Read file
        pfread = c_int()
        self.libstorm.SFileReadFile(fileh, fdata, fsize, byref(pfread), 0)
        self.libstorm.SFileCloseFile(fileh)

        if pfread.value == fsize:
            return fdata.raw
        else:
            return None

    # Writer

    def PutFile(self, fname, buffer, *, cmp1=MPQ_COMP_ZLIB, cmp2=MPQ_COMP_ZLIB):
        if not self.mpqh:
            return None

        # Create temporary file
        f = NamedTemporaryFile(delete=False)
        f.write(bytes(buffer))
        tmpfname = f.name
        f.close()

        try:
            fname = u2b(fname)
        except UnicodeEncodeError:
            fname = u2utf8(fname)

        if platform.system() == "Linux":
            tmpfname = u2b(tmpfname)

        # Add to mpq
        ret = self.libstorm.SFileAddFileEx(
            self.mpqh,
            tmpfname,
            fname,
            MPQ_FILE_COMPRESS | MPQ_FILE_ENCRYPTED | MPQ_FILE_REPLACEEXISTING,
            cmp1,
            cmp2,
        )
        os.unlink(tmpfname)
        return ret

    def PutWave(self, fname, buffer, *, cmp1=MPQ_COMP_ZLIB, cmp2=MPQ_COMP_ZLIB):
        if not self.mpqh:
            return None

        # Create temporary file
        f = NamedTemporaryFile(delete=False)
        f.write(bytes(buffer))
        tmpfname = f.name
        f.close()

        # Add to mpq
        ret = self.libstorm.SFileAddFileEx(
            self.mpqh,
            os.fsencode(tmpfname),
            u2b(fname),
            MPQ_FILE_COMPRESS | MPQ_FILE_ENCRYPTED,
            cmp1,
            cmp2,
        )
        os.unlink(tmpfname)
        return ret

    def GetMaxFileCount(self):
        return self.libstorm.SFileGetMaxFileCount(self.mpqh)

    def SetMaxFileCount(self, count):
        return self.libstorm.SFileSetMaxFileCount(self.mpqh, count)

    def Compact(self):
        return self.libstorm.SFileCompactArchive(self.mpqh, None, 0)

# Convenient functions
def get_chk_from_mpq(mpq_fname):
    mw = MPQ()
    try:
        mw.Open(mpq_fname)
        v = mw.Extract("staredit\\scenario.chk")
        return v
    except (OSError, IOError):
        raise IOError("Failed to open MPQ archive")
    finally:
        mw.Close()

def pack_to_mpq(chk_data, dest_fname):
    mw = MPQ()
    try:
        mw.Create(dest_fname)
        mw.PutFile("staredit\\scenario.chk", chk_data)
    except OSError:
        raise OSError("Failed to pack to MPQ")
    finally:
        mw.Close()

InitMpqLibrary()

if __name__ == "__main__":
    mr = MPQ()
    mr.Open("basemap.scx")
    a = mr.Extract("staredit\\scenario.chk")
    mr.Close()
    print(len(a))

    if os.path.exists("test.scx"):
        os.unlink("test.scx")
    open("test.scx", "wb").write(open("basemap.scx", "rb").read())

    mr.Open("test.scx")
    a = mr.Extract("staredit\\scenario.chk")
    print(len(a))
    mr.PutFile("test", b"1234")
    b = mr.Extract("test")
    mr.Compact()
    print(b)
