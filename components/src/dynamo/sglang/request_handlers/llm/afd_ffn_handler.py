# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
FFN Worker Handler for AFD (Attention-FFN Disaggregation).

AFD separates stateful Attention layers (KV-cache dominated) from 
stateless FFN layers (compute-intensive) during the decode phase.

Architecture: r Attention instances -> 1 shared FFN instance

The FFN worker:
- Receives activations from multiple Attention workers
- Performs FFN computation (compute-bound with sufficient batching)
- Returns results to Attention workers

Reference: https://arxiv.org/abs/2601.21351
"""

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import numpy as np
import torch

import sglang as sgl

from dynamo._core import Component, Context
from dynamo.sglang.args import Config
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler
from dynamo.sglang.afd_communication import (
    AFDCommunicationManager,
    AFDActivationBatch,
    AFDFFNResult,
)
from dynamo.sglang.afd_nixl_transfer import (
    AFDNixlTransferManager,
    AFDTransferConfig,
    AFDActivationBuffer,
    AFDTransferStats,
)
from dynamo.sglang.afd_metrics import AFDMetrics

logger = logging.getLogger(__name__)


class AFDFFNHandler(BaseWorkerHandler):
    """Handler for FFN workers in AFD disaggregated mode.
    
    In AFD mode, the FFN worker is stateless and compute-intensive,
    receiving activations from multiple Attention workers.
    
    Key characteristics:
    - Stateless computation (no KV cache)
    - Becomes compute-bound with sufficient batching
    - Shared by multiple Attention workers (r:1 topology)
    - Aggregates batch from all Attention instances
    """

    def __init__(
        self,
        component: Component,
        engine: sgl.Engine,
        config: Config,
        publisher: DynamoSglangPublisher,
        generate_endpoint=None,
        shutdown_event: Optional[asyncio.Event] = None,
        attention_ratio: int = 1,
    ) -> None:
        """Initialize FFN worker handler for AFD mode.
        
        Args:
            component: The Dynamo runtime component.
            engine: The SGLang engine instance.
            config: SGLang and Dynamo configuration.
            publisher: The SGLang publisher instance.
            generate_endpoint: The endpoint handle for discovery registration.
            shutdown_event: Optional event to signal shutdown.
            attention_ratio: The r in r:1 AFD topology (number of attention workers per FFN).
        """
        super().__init__(
            component, engine, config, publisher, generate_endpoint, shutdown_event
        )
        self.attention_ratio = attention_ratio
        
        # AFD communication
        self._comm_manager: Optional[AFDCommunicationManager] = None
        self._transfer_manager: Optional[AFDNixlTransferManager] = None
        
        # Buffer pool for receiving activations
        self._buffer_pool: asyncio.Queue[AFDActivationBuffer] = asyncio.Queue()
        
        # Pending requests from attention workers
        self._pending_requests: Dict[str, asyncio.Future] = {}
        
        # Batch aggregation
        self._batch_queue: asyncio.Queue[AFDActivationBatch] = asyncio.Queue()
        self._batch_aggregation_task: Optional[asyncio.Task] = None
        
        # Metrics
        self._metrics = AFDMetrics()
        self._transfer_stats = AFDTransferStats()
        
        # FFN computation stats
        self._total_ffn_time_ms = 0.0
        self._total_batches_processed = 0
        self._total_tokens_processed = 0
        
        logger.info(
            f"AFD FFN handler initialized - "
            f"attention_ratio={attention_ratio} (shared by {attention_ratio} Attention workers)"
        )

    async def start_afd_communication(self) -> None:
        """Start AFD communication and batch processing."""
        # Initialize communication manager
        self._comm_manager = AFDCommunicationManager(
            attention_ratio=self.attention_ratio,
            microbatch_size=self.config.dynamo_args.afd_microbatch_size,
            sync_timeout_ms=self.config.dynamo_args.afd_sync_timeout_ms,
        )
        await self._comm_manager.connect()
        
        # Initialize transfer manager
        transfer_config = AFDTransferConfig()
        self._transfer_manager = AFDNixlTransferManager(
            config=transfer_config,
            is_attention_worker=False,  # This is FFN worker
        )
        # Note: FFN worker listens for connections from Attention workers
        await self._transfer_manager.initialize("listen")
        
        # Start batch aggregation task
        self._batch_aggregation_task = asyncio.create_task(self._batch_aggregation_loop())
        
        logger.info("AFD FFN communication started")

    def cleanup(self) -> None:
        """Shutdown the FFN engine and cleanup resources."""
        # Cancel batch aggregation task
        if self._batch_aggregation_task:
            self._batch_aggregation_task.cancel()
        
        # Cancel pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()
        
        # Cleanup transfer manager
        if self._transfer_manager:
            asyncio.create_task(self._transfer_manager.shutdown())
        
        # Disconnect communication manager
        if self._comm_manager:
            asyncio.create_task(self._comm_manager.disconnect())
        
        super().cleanup()
        self.engine.shutdown()
        logger.info("AFD FFN engine shutdown")

    async def generate(
        self, request: Dict[str, Any], context: Context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process FFN computation for activations from Attention workers.
        
        In AFD mode, the FFN worker:
        1. Receives activations from Attention workers
        2. Aggregates batch for efficient computation
        3. Performs FFN matrix multiplications (compute-bound)
        4. Returns results to respective Attention workers
        
        Args:
            request: Request dict containing activations from Attention worker.
            context: Context object for cancellation handling.
            
        Yields:
            Response dicts with FFN computation results.
        """
        request_id = context.id()
        logger.debug(f"AFD FFN Request ID: {request_id}")
        
        # Check if this is an activation request from Attention worker
        if "activations" in request:
            # Process as activation request
            async for out in self._process_activation_request(request, context):
                yield out
        else:
            # Unknown request type
            logger.warning(f"Unknown request type for FFN worker: {list(request.keys())}")
            yield {
                "error": "Invalid request type for FFN worker",
                "meta_info": {"afd_mode": "ffn"},
            }

    async def _process_activation_request(
        self,
        request: Dict[str, Any],
        context: Context,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process an activation request from an Attention worker.
        
        Args:
            request: Request containing activations
            context: Request context
            
        Yields:
            FFN computation results
        """
        request_id = context.id()
        
        # Extract activation data
        activations = request.get("activations")
        layer_idx = request.get("layer_idx", 0)
        metadata = request.get("metadata", {})
        
        if activations is None:
            yield {"error": "No activations in request", "request_id": request_id}
            return
        
        ffn_start = time.perf_counter()
        
        # Convert to tensor if needed
        if isinstance(activations, np.ndarray):
            activations_tensor = torch.from_numpy(activations).cuda()
        elif isinstance(activations, torch.Tensor):
            activations_tensor = activations.cuda()
        else:
            yield {"error": f"Unsupported activation type: {type(activations)}"}
            return
        
        # Process through FFN layers
        # In full AFD, this would use the SGLang engine's FFN-only mode
        try:
            # For now, use the engine's generate as a placeholder
            # TODO: Implement FFN-only forward pass
            
            # Simulate FFN computation
            output = await self._run_ffn_forward(activations_tensor, layer_idx)
            
            ffn_time = (time.perf_counter() - ffn_start) * 1000
            self._total_ffn_time_ms += ffn_time
            self._total_batches_processed += 1
            self._total_tokens_processed += activations_tensor.shape[0] * activations_tensor.shape[1]
            
            # Update metrics
            self._metrics.record_ffn_computation(
                batch_size=activations_tensor.shape[0],
                seq_len=activations_tensor.shape[1],
                layer_idx=layer_idx,
                latency_ms=ffn_time,
            )
            
            # Yield result
            yield {
                "output": output.cpu().numpy() if isinstance(output, torch.Tensor) else output,
                "request_id": request_id,
                "layer_idx": layer_idx,
                "ffn_time_ms": ffn_time,
                "meta_info": {
                    "afd_mode": "ffn",
                    "attention_ratio": self.attention_ratio,
                    "batch_size": activations_tensor.shape[0],
                },
            }
            
        except Exception as e:
            logger.error(f"FFN computation failed: {e}")
            yield {"error": str(e), "request_id": request_id}

    async def _run_ffn_forward(
        self,
        activations: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Run FFN forward pass.
        
        In full AFD, this would use SGLang's FFN-only mode.
        For now, we simulate the computation.
        
        Args:
            activations: Input activations [batch, seq, hidden]
            layer_idx: Transformer layer index
            
        Returns:
            FFN output [batch, seq, hidden]
        """
        # Simulate FFN computation
        # In production, this would call the SGLang engine's FFN layers
        
        # Simple placeholder: identity + noise
        await asyncio.sleep(0.001)  # Simulate compute time
        
        return activations

    async def _batch_aggregation_loop(self) -> None:
        """Background task for aggregating activation batches.
        
        This loop collects activation requests from multiple Attention workers
        and batches them for efficient FFN computation.
        """
        logger.info("AFD batch aggregation loop started")
        
        batch_timeout_ms = 10  # 10ms batch timeout
        max_batch_size = 64  # Maximum batch size
        
        while True:
            try:
                batch: List[AFDActivationBatch] = []
                deadline = asyncio.get_event_loop().time() + batch_timeout_ms / 1000
                
                # Collect batch within timeout window
                while len(batch) < max_batch_size:
                    remaining_time = deadline - asyncio.get_event_loop().time()
                    if remaining_time <= 0:
                        break
                    
                    try:
                        activation = await asyncio.wait_for(
                            self._batch_queue.get(),
                            timeout=remaining_time,
                        )
                        batch.append(activation)
                    except asyncio.TimeoutError:
                        break
                
                if not batch:
                    continue
                
                # Process batch
                logger.debug(f"Processing batch of {len(batch)} activations")
                
                # Aggregate activations
                # In production, this would run actual FFN computation
                
                # Send results back
                for activation in batch:
                    result = AFDFFNResult(
                        request_id=activation.request_id,
                        output=activation.activations,  # Placeholder
                    )
                    if self._comm_manager:
                        self._comm_manager.handle_ffn_result(result)
                
                self._total_batches_processed += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch aggregation error: {e}")
                await asyncio.sleep(0.1)
        
        logger.info("AFD batch aggregation loop stopped")

    async def aggregate_batch(
        self, requests: list[Dict[str, Any]], timeout_ms: int = 100
    ) -> Dict[str, Any]:
        """Aggregate activations from multiple Attention workers.
        
        This method collects activations from Attention workers within
        a timeout window to form an efficient batch for FFN computation.
        
        Args:
            requests: List of activation requests from Attention workers.
            timeout_ms: Maximum time to wait for batch aggregation.
            
        Returns:
            Aggregated batch ready for FFN computation.
        """
        aggregated = {
            "activations": [],
            "request_ids": [],
            "batch_size": 0,
        }
        
        deadline = asyncio.get_event_loop().time() + timeout_ms / 1000
        
        for req in requests:
            if asyncio.get_event_loop().time() > deadline:
                break
            aggregated["activations"].append(req.get("activations"))
            aggregated["request_ids"].append(req.get("request_id"))
            aggregated["batch_size"] += 1
        
        logger.debug(f"Aggregated batch size: {aggregated['batch_size']}")
        return aggregated

    def get_metrics(self) -> Dict[str, Any]:
        """Get AFD FFN-specific metrics."""
        return {
            "total_ffn_time_ms": self._total_ffn_time_ms,
            "total_batches_processed": self._total_batches_processed,
            "total_tokens_processed": self._total_tokens_processed,
            "avg_ffn_time_ms": self._total_ffn_time_ms / max(1, self._total_batches_processed),
            "attention_ratio": self.attention_ratio,
            "transfer_stats": self._transfer_stats.to_dict(),
        }
