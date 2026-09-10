from __future__ import annotations

import os
from typing import List, Union
from tdw.controller import Controller
from requests import get
from pathlib import Path
import shutil
import hashlib
import platform
import re
from urllib.parse import urlparse

class AssetCachedController(Controller):
    def __init__(self, cache_dir="transport_challenge_asset_bundles", **kwargs):
        self.cache_dir = None
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir).expanduser().resolve()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(**kwargs)
    
    def communicate(self, commands: dict | List[dict]):
        '''
        override and calculate add_ons' commands in advance to make cache of asset
        '''
        if self.cache_dir is None:
            return super().communicate(commands)
        
        if isinstance(commands, dict):
            commands = [commands]
        add_ons_t = self.add_ons
        self.add_ons = [] # remove add_ons
        for m in add_ons_t:
            if not m.initialized:
                commands.extend(m.get_initialization_commands())
                m.initialized = True
            else:
                commands.extend(m.commands)
                m.commands.clear()
        for m in add_ons_t:
            m.before_send(commands)
        
        for cmd in commands:
            if "url" in cmd:
                cmd["url"] = self.get_asset(cmd["url"])
        resp = super().communicate(commands)
        for m in add_ons_t:
            m.on_send(resp)
        self.add_ons = add_ons_t # resume add_ons
        return resp
    
    @staticmethod
    def normalize_asset_url(url: str) -> str:
        """Select the asset-bundle URL compatible with the active player."""
        parsed_url = urlparse(url)
        if (platform.system() == "Windows"
                and parsed_url.hostname == "tdw-public.s3.amazonaws.com"
                and "/linux/" in parsed_url.path.lower()):
            return re.sub(r"/linux/", "/windows/", url, flags=re.IGNORECASE)
        return url

    def get_asset(self, url: str):
        url = self.normalize_asset_url(url)
        parsed_url = urlparse(url)
        # Local catalogs use file:/// URIs. They already point to verified files
        # under TDW_ASSET_CACHE_DIR and must not be sent through requests.
        if parsed_url.scheme.lower() == "file":
            return url

        name = hashlib.md5(url.encode()).hexdigest()
        path = self.cache_dir.joinpath(name)
        if not path.is_file() or path.stat().st_size == 0:
            print(f"downloading {url} to {str(path)}")
            response = get(url, timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
        # Path.as_uri() produces a valid file:///D:/... URI on Windows.
        return path.resolve().as_uri()
    
    def clear_cache(self):
        shutil.rmtree(self.cache_dir.resolve())
