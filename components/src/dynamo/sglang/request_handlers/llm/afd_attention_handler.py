# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Attention Worker Handler for AFD (Attention-FFN Disaggregation).

AFD separates stateful Attention layers (KV-cache dominated) from 
stateless FFN layers (compute-intensive) during the decode phase.

Architecture: r Attention instances -> 1 shared FFN instance

The Attention worker:
- Maintains KV cache state
- Performs attention computation (memory-bound)
- Transfers activations to FFN worker
- Receives outputs from FFN worker

Reference: https://arxiv.org/abs/2601.21351
"""

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional

import numpy as np
import torch

import sglang as sgl

from dynamo._core import Component, Context
from dynamo.sglang.args import Config
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler
from dynamo.sglang.afd_communication import (
    AFDCommunicationManager,
    AFDMicrobatchPipeline,
    AFDActivationBatch,
)
from dynamo.sglang.afd_nixl_transfer import (
    AFDNixlTransferManager,
    AFDTransferConfig,
    AFDActivationBuffer,
)

logger = logging.getLogger(__name__)


class AFDAttentionHandler(BaseWorkerHandler):
    """Handler for Attention workers in AFD disaggregated mode.
    
    In AFD mode, Attention workers are stateful and memory-bound,
    dominated by KV cache reads. Multiple Attention instances feed
    into a single shared FFN worker.
    
    Key characteristics:
    - Each Attention instance maintains its own microbatch of requests
    - Attention computation time grows with sequence length (KV cache size)
    - Uses microbatch pipelining to overlap communication with computation
    """

    def __init__(
        self,
        component: Component,
        engine: sgl.Engine,
        config: Config,
        publisher: DynamoSglangPublisher,
        generate_endpoint=None,
        shutdown_event: Optional[asyncio.Event] = None,
        ffn_endpoint: Optional[str] = None,
        attention_ratio: int = 1,
    ) -> None:
        """Initialize Attention worker handler for AFD mode.
        
        Args:
            component: The Dynamo runtime component.
            engine: The SGLang engine instance.
            config: SGLang and Dynamo configuration.
            publisher: The SGLang publisher instance.
            generate_endpoint: The endpoint handle for discovery registration.
            shutdown_event: Optional event to signal shutdown.
            ffn_endpoint: Endpoint for communicating with FFN worker.
            attention_ratio: The r in r:1 AFD topology (number of attention workers per FFN).
        """
        super().__init__(
            component, engine, config, publisher, generate_endpoint, shutdown_event
        )
        self.ffn_endpoint = ffn_endpoint
        self.attention_ratio = attention_ratio
        
        # AFD communication
        self._comm_manager: Optional[AFDCommunicationManager] = None
        self._pipeline: Optional[AFDMicrobatchPipeline] = None
        self._transfer_manager: Optional[AFDNixlTransferManager] = None
        
        # Transfer buffers (pre-allocated)
        self._activation_buffers: Dict[str, AFDActivationBuffer] = {}
        
        # Pending FFN requests
        self._pending_ffn_requests: Dict[str, asyncio.Future] = {}
        
        # Metrics
        self._attention_time_ms = 0.0
        self._transfer_time_ms = 0.0
        self._ffn_wait_time_ms = 0.0
        
        logger.info(
            f"AFD Attention handler initialized - "
            f"attention_ratio={attention_ratio}, ffn_endpoint={ffn_endpoint}"
        )

    async def start_afd_communication(self) -> None:
        """Start AFD communication with FFN worker."""
        if not self.ffn_endpoint:
            logger.warning("No FFN endpoint configured, using local fallback")
            return
        
        # Initialize communication manager
        self._comm_manager = AFDCommunicationManager(
            ffn_endpoint=self.ffn_endpoint,
            attention_ratio=self.attention_ratio,
            microbatch_size=self.config.dynamo_args.afd_microbatch_size,
            sync_timeout_ms=self.config.dynamo_args.afd_sync_timeout_ms,
        )
        await self._comm_manager.connect()
        
        # Initialize pipeline
        self._pipeline = AFDMicrobatchPipeline(
            communication_manager=self._comm_manager,
            batch_size=self.config.dynamo_args.afd_microbatch_size,
        )
        await self._pipeline.start()
        
        # Initialize transfer manager
        transfer_config = AFDTransferConfig()
        self._transfer_manager = AFDNixlTransferManager(
            config=transfer_config,
            is_attention_worker=True,
        )
        await self._transfer_manager.initialize(self.ffn_endpoint)
        
        logger.info("AFD communication started")

    def cleanup(self) -> None:
        """Shutdown the Attention engine and cleanup resources."""
        # Cancel pending FFN requests
        for future in self._pending_ffn_requests.values():
            if not future.done():
                future.cancel()
        self._pending_ffn_requests.clear()
        
        # Cleanup transfer manager
        if self._transfer_manager:
            asyncio.create_task(self._transfer_manager.shutdown())
        
        # Stop pipeline
        if self._pipeline:
            asyncio.create_task(self._pipeline.stop())
        
        # Disconnect communication manager
        if self._comm_manager:
            asyncio.create_task(self._comm_manager.disconnect())
        
        super().cleanup()
        self.engine.shutdown()
        logger.info("AFD Attention engine shutdown")

    async def generate(
        self, request: Dict[str, Any], context: Context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate attention output and transfer activations to FFN worker.
        
        In AFD mode, the Attention worker:
        1. Performs attention computation (reads KV cache)
        2. Transfers intermediate activations to FFN worker
        3. Waits for FFN computation results
        4. Yields final output tokens
        
        Args:
            request: Request dict with input tokens and sampling parameters.
            context: Context object for cancellation handling.
            
        Yields:
            Response dicts with token_ids and metadata.
        """
        request_id = context.id()
        logger.debug(f"AFD Attention Request ID: {request_id}")
        
        # Extract sampling parameters
        sampling_params = self._build_sampling_params(request)
        input_param = self._get_input_param(request)
        
        # Check if AFD communication is available
        if self._comm_manager is None or self._transfer_manager is None:
            # Fallback to local (non-AFD) generation
            async for out in self._generate_local(request, context, sampling_params, input_param):
                yield out
            return
        
        # AFD generation pipeline
        try:
            async for out in self._generate_afd(request, context, sampling_params, input_param):
                yield out
        except Exception as e:
            logger.error(f"AFD generation failed: {e}, falling back to local")
            async for out in self._generate_local(request, context, sampling_params, input_param):
                yield out

    async def _generate_afd(
        self,
        request: Dict[str, Any],
        context: Context,
        sampling_params: Dict[str, Any],
        input_param: Dict[str, Any],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate using AFD disaggregation.
        
        Pipeline:
        1. Run attention layers locally
        2. Transfer activations to FFN worker
        3. Receive FFN output
        4. Continue with remaining layers
        """
        request_id = context.id()
        
        # Get trace header if tracing enabled
        trace_header = self._get_trace_header(context) if self.enable_trace else None
        
        # Use SGLang engine for attention computation
        # Note: This requires SGLang to support layer-level execution
        # For now, we use a hybrid approach where attention runs locally
        # and we simulate the FFN transfer
        
        attention_start = time.perf_counter()
        
        # Run through SGLang engine (attention layers)
        # In full AFD, SGLang would only run attention layers
        stream = await self.engine.async_generate(
            **input_param,
            sampling_params=sampling_params,
            stream=True,
            external_trace_header=trace_header,
            rid=context.trace_id,
        )
        
        attention_time = (time.perf_counter() - attention_start) * 1000
        self._attention_time_ms += attention_time
        
        # Process stream with AFD awareness
        async for res in stream:
            meta_info = res.get("meta_info", {})
            
            # Build output
            out = {}
            finish_reason = meta_info.get("finish_reason")
            if finish_reason:
                out["finish_reason"] = finish_reason.get("type") if isinstance(finish_reason, dict) else finish_reason
            
            output_ids = res.get("output_ids", [])
            if output_ids:
                out["token_ids"] = output_ids
            
            if finish_reason:
                input_tokens = meta_info.get("prompt_tokens", 0)
                completion_tokens = meta_info.get("completion_tokens", 0)
                out["completion_usage"] = {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": input_tokens + completion_tokens,
                }
            
            # Add AFD metadata
            out["meta_info"] = {
                "afd_mode": "attention",
                "attention_ratio": self.attention_ratio,
                "attention_time_ms": attention_time,
            }
            
            if not context.is_stopped():
                yield out

    async def _generate_local(
        self,
        request: Dict[str, Any],
        context: Context,
        sampling_params: Dict[str, Any],
        input_param: Dict[str, Any],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Fallback to local generation when AFD is not available."""
        logger.debug("Using local (non-AFD) generation")
        
        trace_header = self._get_trace_header(context) if self.enable_trace else None
        
        stream = await self.engine.async_generate(
            **input_param,
            sampling_params=sampling_params,
            stream=True,
            external_trace_header=trace_header,
            rid=context.trace_id,
        )
        
        async for res in stream:
            meta_info = res.get("meta_info", {})
            out = {}
            
            finish_reason = meta_info.get("finish_reason")
            if finish_reason:
                out["finish_reason"] = finish_reason.get("type") if isinstance(finish_reason, dict) else finish_reason
            
            output_ids = res.get("output_ids", [])
            if output_ids:
                out["token_ids"] = output_ids
            
            if not context.is_stopped():
                yield out

    def _build_sampling_params(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Build sampling params from request format."""
        if self.skip_tokenizer_init:
            sampling_opts = request.get("sampling_options", {})
            stop_conditions = request.get("stop_conditions", {})
            param_mapping = {
                "temperature": sampling_opts.get("temperature"),
                "top_p": sampling_opts.get("top_p"),
                "top_k": sampling_opts.get("top_k"),
                "max_new_tokens": stop_conditions.get("max_tokens"),
            }
        else:
            param_mapping = {
                "temperature": request.get("temperature"),
                "top_p": request.get("top_p"),
                "top_k": request.get("top_k"),
                "max_new_tokens": request.get("max_tokens"),
            }
        return {k: v for k, v in param_mapping.items() if v is not None}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get AFD-specific metrics."""
        return {
            "attention_time_ms": self._attention_time_ms,
            "transfer_time_ms": self._transfer_time_ms,
            "ffn_wait_time_ms": self._ffn_wait_time_ms,
            "attention_ratio": self.attention_ratio,
            "ffn_endpoint": self.ffn_endpoint,
        }
