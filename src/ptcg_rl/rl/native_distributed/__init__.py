"""In-memory contracts and transport for distributed native rollout."""

from ptcg_rl.rl.native_distributed.array_schema import (
    COMPACT_FRAGMENT_ARRAY_FIELDS,
    SEQUENCE_COMPACT_FRAGMENT_ARRAY_FIELDS,
)
from ptcg_rl.rl.native_distributed.codec import (
    DecodedNativeRolloutPart,
    decode_compact_fragment_part,
    encode_compact_fragment_part,
    recv_compact_fragment_part,
    send_compact_fragment_part,
)
from ptcg_rl.rl.native_distributed.contracts import (
    NativeRolloutLeaseIdentity,
    NativeRolloutPartIdentity,
    NativeRolloutWindowIdentity,
    NativeRolloutWorkerIdentity,
)
from ptcg_rl.rl.native_distributed.handshake import (
    decode_worker_hello,
    decode_worker_ready,
    encode_worker_hello,
    encode_worker_ready,
    recv_worker_hello,
    recv_worker_ready,
    send_worker_hello,
    send_worker_ready,
)
from ptcg_rl.rl.native_distributed.session import (
    NativeRolloutPartReceiver,
    NativeRolloutPartSink,
    NativeRolloutProtocolError,
    NativeRolloutRemoteAbortError,
    NativeRolloutSessionError,
    NativeRolloutTimeoutError,
)
from ptcg_rl.rl.native_distributed.weights import (
    BehaviorPolicyTensorSpec,
    DecodedBehaviorPolicyWeights,
    behavior_policy_artifact_fingerprint,
    behavior_policy_tensor_specs,
    decode_behavior_policy_weights,
    encode_behavior_policy_weights,
    recv_behavior_policy_weights,
    send_behavior_policy_weights,
)

__all__ = [
    "BehaviorPolicyTensorSpec",
    "COMPACT_FRAGMENT_ARRAY_FIELDS",
    "SEQUENCE_COMPACT_FRAGMENT_ARRAY_FIELDS",
    "DecodedBehaviorPolicyWeights",
    "DecodedNativeRolloutPart",
    "NativeRolloutLeaseIdentity",
    "NativeRolloutPartReceiver",
    "NativeRolloutPartSink",
    "NativeRolloutPartIdentity",
    "NativeRolloutProtocolError",
    "NativeRolloutRemoteAbortError",
    "NativeRolloutSessionError",
    "NativeRolloutTimeoutError",
    "NativeRolloutWindowIdentity",
    "NativeRolloutWorkerIdentity",
    "behavior_policy_artifact_fingerprint",
    "behavior_policy_tensor_specs",
    "decode_behavior_policy_weights",
    "decode_compact_fragment_part",
    "decode_worker_hello",
    "decode_worker_ready",
    "encode_behavior_policy_weights",
    "encode_compact_fragment_part",
    "encode_worker_hello",
    "encode_worker_ready",
    "recv_behavior_policy_weights",
    "recv_compact_fragment_part",
    "recv_worker_hello",
    "recv_worker_ready",
    "send_behavior_policy_weights",
    "send_compact_fragment_part",
    "send_worker_hello",
    "send_worker_ready",
]
