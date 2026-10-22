#ifndef KATAGO_WEB_BRIDGE_ABI_H_
#define KATAGO_WEB_BRIDGE_ABI_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KATAGO_BRIDGE_ABI_VERSION 1

// Slot state machine enum
typedef enum {
  KATAGO_SLOT_FREE = 0,
  KATAGO_SLOT_WRITING = 1,
  KATAGO_SLOT_READY = 2,
  KATAGO_SLOT_RUNNING = 3,
  KATAGO_SLOT_DONE = 4,
  KATAGO_SLOT_ERROR = 5
} KatagoSlotStatus;

// Error codes
typedef enum {
  KATAGO_BRIDGE_OK = 0,
  KATAGO_BRIDGE_ERR_INVALID_DESCRIPTOR = 1,
  KATAGO_BRIDGE_ERR_UNSUPPORTED_VERSION = 2,
  KATAGO_BRIDGE_ERR_STALE_GENERATION = 3,
  KATAGO_BRIDGE_ERR_CANCELLED = 4,
  KATAGO_BRIDGE_ERR_INFERENCE_FAILED = 5,
  KATAGO_BRIDGE_ERR_BUFFER_OVERFLOW = 6,
  KATAGO_BRIDGE_ERR_DEVICE_LOST = 7
} KatagoBridgeErrorCode;

// Tensor Layout enum
typedef enum {
  KATAGO_LAYOUT_NCHW = 0,
  KATAGO_LAYOUT_NHWC = 1
} KatagoTensorLayout;

// Exact fixed-width descriptor layout (little-endian, wasm32 compatible)
#pragma pack(push, 4)
typedef struct {
  // Header (bytes 0..31)
  uint32_t abi_version;            // 0: must equal KATAGO_BRIDGE_ABI_VERSION (1)
  uint32_t engine_generation;       // 4: incremented on engine reset/model reload
  uint64_t request_sequence;       // 8: monotonically increasing sequence number
  uint32_t slot_id;                // 16: slot index (e.g. 0 or 1)
  uint32_t model_id;               // 20: loaded model identifier
  uint32_t status;                 // 24: KatagoSlotStatus (Atomics control word)
  uint32_t cancel_generation;      // 28: if cancel_generation >= engine_generation, cancel requested

  // Error Info (bytes 32..47)
  uint32_t error_code;             // 32: KatagoBridgeErrorCode
  uint32_t error_msg_offset;       // 36: wasm memory offset to utf-8 error string
  uint32_t error_msg_capacity;     // 40: capacity of error string buffer
  uint32_t reserved0;              // 44: padding / alignment

  // Batch & Board Dimensions (bytes 48..79)
  uint32_t batch_size;             // 48: actual number of filled rows
  uint32_t max_batch_size;         // 52: maximum allocated batch capacity
  uint32_t nn_x_len;               // 56: board width (e.g. 19)
  uint32_t nn_y_len;               // 60: board height (e.g. 19)
  uint32_t num_spatial_channels;   // 64: spatial feature channels (e.g. 22)
  uint32_t num_global_channels;    // 68: global feature channels (e.g. 19)
  uint32_t num_meta_channels;      // 72: metadata channels (e.g. 0 or SGFMetadata)
  uint32_t num_policy_channels;    // 76: policy channels (e.g. 2)

  // Channels & Layout (bytes 80..95)
  uint32_t num_value_channels;        // 80: value channels (e.g. 3)
  uint32_t num_score_value_channels;  // 84: score value channels (e.g. 6)
  uint32_t num_ownership_channels;    // 88: ownership channels (e.g. 1)
  uint32_t spatial_layout;            // 92: KatagoTensorLayout (NCHW = 0)

  // Input Tensors (bytes 96..127)
  uint32_t mask_offset;            // 96
  uint32_t mask_bytes;             // 100
  uint32_t spatial_offset;         // 104
  uint32_t spatial_bytes;          // 108
  uint32_t global_offset;          // 112
  uint32_t global_bytes;           // 116
  uint32_t meta_offset;            // 120
  uint32_t meta_bytes;             // 124

  // Output Tensors (bytes 128..167)
  uint32_t policy_pass_offset;     // 128
  uint32_t policy_pass_bytes;      // 132
  uint32_t policy_offset;          // 136
  uint32_t policy_bytes;           // 140
  uint32_t value_offset;           // 144
  uint32_t value_bytes;            // 148
  uint32_t score_value_offset;     // 152
  uint32_t score_value_bytes;      // 156
  uint32_t ownership_offset;       // 160
  uint32_t ownership_bytes;        // 164

  // Total size: 168 bytes
} BridgeSlotDescriptor;
#pragma pack(pop)

#ifdef __cplusplus
static_assert(sizeof(BridgeSlotDescriptor) == 168, "BridgeSlotDescriptor size must be exactly 168 bytes");
static_assert(offsetof(BridgeSlotDescriptor, abi_version) == 0, "abi_version offset");
static_assert(offsetof(BridgeSlotDescriptor, engine_generation) == 4, "engine_generation offset");
static_assert(offsetof(BridgeSlotDescriptor, request_sequence) == 8, "request_sequence offset");
static_assert(offsetof(BridgeSlotDescriptor, slot_id) == 16, "slot_id offset");
static_assert(offsetof(BridgeSlotDescriptor, model_id) == 20, "model_id offset");
static_assert(offsetof(BridgeSlotDescriptor, status) == 24, "status offset");
static_assert(offsetof(BridgeSlotDescriptor, cancel_generation) == 28, "cancel_generation offset");
static_assert(offsetof(BridgeSlotDescriptor, error_code) == 32, "error_code offset");
static_assert(offsetof(BridgeSlotDescriptor, error_msg_offset) == 36, "error_msg_offset offset");
static_assert(offsetof(BridgeSlotDescriptor, error_msg_capacity) == 40, "error_msg_capacity offset");
static_assert(offsetof(BridgeSlotDescriptor, batch_size) == 48, "batch_size offset");
static_assert(offsetof(BridgeSlotDescriptor, max_batch_size) == 52, "max_batch_size offset");
static_assert(offsetof(BridgeSlotDescriptor, nn_x_len) == 56, "nn_x_len offset");
static_assert(offsetof(BridgeSlotDescriptor, nn_y_len) == 60, "nn_y_len offset");
static_assert(offsetof(BridgeSlotDescriptor, num_spatial_channels) == 64, "num_spatial_channels offset");
static_assert(offsetof(BridgeSlotDescriptor, num_global_channels) == 68, "num_global_channels offset");
static_assert(offsetof(BridgeSlotDescriptor, num_meta_channels) == 72, "num_meta_channels offset");
static_assert(offsetof(BridgeSlotDescriptor, num_policy_channels) == 76, "num_policy_channels offset");
static_assert(offsetof(BridgeSlotDescriptor, num_value_channels) == 80, "num_value_channels offset");
static_assert(offsetof(BridgeSlotDescriptor, num_score_value_channels) == 84, "num_score_value_channels offset");
static_assert(offsetof(BridgeSlotDescriptor, num_ownership_channels) == 88, "num_ownership_channels offset");
static_assert(offsetof(BridgeSlotDescriptor, spatial_layout) == 92, "spatial_layout offset");
static_assert(offsetof(BridgeSlotDescriptor, mask_offset) == 96, "mask_offset offset");
static_assert(offsetof(BridgeSlotDescriptor, spatial_offset) == 104, "spatial_offset offset");
static_assert(offsetof(BridgeSlotDescriptor, global_offset) == 112, "global_offset offset");
static_assert(offsetof(BridgeSlotDescriptor, meta_offset) == 120, "meta_offset offset");
static_assert(offsetof(BridgeSlotDescriptor, policy_pass_offset) == 128, "policy_pass_offset offset");
static_assert(offsetof(BridgeSlotDescriptor, policy_offset) == 136, "policy_offset offset");
static_assert(offsetof(BridgeSlotDescriptor, value_offset) == 144, "value_offset offset");
static_assert(offsetof(BridgeSlotDescriptor, score_value_offset) == 152, "score_value_offset offset");
static_assert(offsetof(BridgeSlotDescriptor, ownership_offset) == 160, "ownership_offset offset");
#endif

#ifdef __cplusplus
}
#endif

#endif // KATAGO_WEB_BRIDGE_ABI_H_
