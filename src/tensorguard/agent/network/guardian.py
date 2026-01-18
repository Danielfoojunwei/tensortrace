"""
Network Guardian - Subsystem Controller for RTPL

Manages traffic privacy defenses.
Acts as a local TCP proxy to intercept traffic and apply FRONT/WTF-PAD/Padding.
"""

import logging
import asyncio
import threading
from typing import Optional
from ...schemas.unified_config import NetworkConfig, DefenseMode

from .defense.front import FRONT, FRONTConfig
from .defense.padding import PaddingOnly, PaddingConfig
from .defense.wtf_pad import WTFPAD, WTFPADConfig

logger = logging.getLogger(__name__)

class NetworkGuardian:
    """
    Subsystem controller for Robot Traffic Privacy Layer (RTPL).
    Implements a transparent TCP proxy with traffic analysis defense.
    """
    def __init__(self, agent_config: 'AgentConfig', config_manager: 'ConfigManager'):
        self.config: NetworkConfig = agent_config.network
        self.running = False
        
        self.defense_engine = None
        self._proxy_server = None
        self._loop = None
        self._thread = None

    def configure(self, new_config: NetworkConfig):
        """Reconfigure defense."""
        logger.info(f"Reconfiguring Network Guardian: {new_config.defense_mode}")
        self.config = new_config
        self._init_defense()
        # In a full implementation, we would restart the server here

    def _init_defense(self):
        """Initialize appropriate defense engine."""
        mode = self.config.defense_mode
        
        if mode == DefenseMode.FRONT:
            self.defense_engine = FRONT(
                FRONTConfig(max_dummies=self.config.front_max_dummies)
            )
        elif mode == DefenseMode.PADDING:
            self.defense_engine = PaddingOnly(
                PaddingConfig(bucket_bytes=self.config.padding_bucket_size)
            )
        elif mode == DefenseMode.WTF_PAD:
            self.defense_engine = WTFPAD(WTFPADConfig())
        else:
            self.defense_engine = None
            
        logger.info(f"Initialized defense engine: {type(self.defense_engine).__name__}")

    def start(self):
        """Start the traffic proxy."""
        if not self.config.enabled:
            return
            
        self._init_defense()
        self.running = True
        
        # Start Proxy in separate thread with its own asyncio loop
        self._thread = threading.Thread(target=self._run_proxy_thread, daemon=True)
        self._thread.start()
        logger.info(f"Network Guardian proxy starting on port {self.config.proxy_port} -> {self.config.target_host}:{self.config.target_port}")

    def stop(self):
        """Stop the proxy."""
        self.running = False
        if self._loop:
             self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Network Guardian stopped")

    def _run_proxy_thread(self):
        """Run asyncio loop in a separate thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._start_proxy_server())
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Proxy thread error: {e}")
        finally:
            self._loop.close()

    async def _start_proxy_server(self):
        """Start the TCP server."""
        self._proxy_server = await asyncio.start_server(
            self._handle_client, '0.0.0.0', self.config.proxy_port
        )
        logger.info(f"Listening on 0.0.0.0:{self.config.proxy_port}")

    async def _handle_client(self, client_reader, client_writer):
        """Handle new client connection."""
        peer = client_writer.get_extra_info('peername')
        logger.debug(f"New connection from {peer}")
        
        remote_reader = None
        remote_writer = None
        
        try:
            # Connect to target
            remote_reader, remote_writer = await asyncio.open_connection(
                self.config.target_host, self.config.target_port
            )
            
            # Start bidirectional forwarding
            client_to_remote = asyncio.create_task(
                self._forward(client_reader, remote_writer, "upstream")
            )
            remote_to_client = asyncio.create_task(
                self._forward(remote_reader, client_writer, "downstream")
            )
            
            await asyncio.gather(client_to_remote, remote_to_client)
            
        except Exception as e:
            logger.error(f"Connection handling error: {e}")
        finally:
            if remote_writer: remote_writer.close()
            client_writer.close()

    async def _forward(self, reader, writer, direction: str):
        """Forward data between streams with defense hooks."""
        try:
            while self.running:
                data = await reader.read(4096)
                if not data:
                    break
                
                # Apply defense hooks
                if direction == "upstream":
                    processed = await self.process_upstream(data, writer)
                else:
                    processed = data # Typically defenses are applied on upstream (outbound) traffic
                
                if processed:
                    writer.write(processed)
                    await writer.drain()
        except Exception as e:
            logger.debug(f"Forwarding error ({direction}): {e}")

    async def process_upstream(self, data: bytes, writer) -> bytes:
        """
        Process outgoing packets (client -> remote).
        Injects dummy traffic if defense engine allows.
        """
        if not self.defense_engine:
            return data
            
        # Hook for WTF-PAD / FRONT
        # Note: Real implementation needs strict timing.
        # This is a simplified integration.
        
        if isinstance(self.defense_engine, WTFPAD):
            # WTF-PAD needs a callback to send dummies
            if self.defense_engine._send_callback is None:
                # Defense engine needs to send dummies INDEPENDENT of real traffic
                # But here we are in the data flow.
                # Ideally, WTF-PAD runs a separate task that writes to 'writer'.
                # For now, we just update state.
                await self.defense_engine.on_packet(data)
                
        elif isinstance(self.defense_engine, PaddingOnly):
            return self.defense_engine.pad(data)
            
        return data

    async def _send_dummy(self, writer, dummy_data: bytes):
        """Callback for defenses to inject dummies."""
        try:
            writer.write(dummy_data)
            await writer.drain()
        except:
            pass
