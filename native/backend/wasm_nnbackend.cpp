#include "neuralnet/nninterface.h"
#include "neuralnet/nneval.h"
#include "neuralnet/nninputs.h"
#include "neuralnet/modelversion.h"
#include "../bridge/bridge_abi.h"

#include "external/nlohmann_json/json.hpp"
#include <emscripten.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cassert>

using namespace std;
using json = nlohmann::json;

#define KATAGO_MAX_SLOTS 8

// Global slots storage for JSPI bridge
static BridgeSlotDescriptor g_slot_descriptors[KATAGO_MAX_SLOTS];

// Static assertions ensuring ABI compatibility
static_assert(sizeof(BridgeSlotDescriptor) == 168, "BridgeSlotDescriptor size mismatch");
static_assert(KATAGO_BRIDGE_ABI_VERSION == 1, "KATAGO_BRIDGE_ABI_VERSION mismatch");

class SlotAllocator {
private:
  std::mutex mutex;
  std::condition_variable cv;
  bool inUse[KATAGO_MAX_SLOTS];
  uint64_t sequenceCounter;
  uint32_t currentGeneration;

public:
  SlotAllocator() : sequenceCounter(1), currentGeneration(1) {
    for(int i = 0; i < KATAGO_MAX_SLOTS; i++) {
      inUse[i] = false;
    }
  }

  void setGeneration(uint32_t gen) {
    std::lock_guard<std::mutex> lock(mutex);
    currentGeneration = gen;
    cv.notify_all();
  }

  int acquireSlot(uint64_t& outSequence, uint32_t& outGeneration) {
    std::unique_lock<std::mutex> lock(mutex);
    while(true) {
      for(int i = 0; i < KATAGO_MAX_SLOTS; i++) {
        if(!inUse[i]) {
          inUse[i] = true;
          outSequence = ++sequenceCounter;
          outGeneration = currentGeneration;
          return i;
        }
      }
      cv.wait(lock);
    }
  }

  void releaseSlot(int slotId) {
    std::lock_guard<std::mutex> lock(mutex);
    if(slotId >= 0 && slotId < KATAGO_MAX_SLOTS) {
      inUse[slotId] = false;
      cv.notify_one();
    }
  }
};

static SlotAllocator g_slotAllocator;

extern "C" {
  // Suspending JSPI import: evaluator thread calls this, Wasm suspends, JS coordinator executes ORT inference,
  // then resolves promise, and evaluator thread resumes!
  extern int js_eval_batch_async(int slot_id, uint32_t generation, uint32_t seq_low, uint32_t seq_high);

  EMSCRIPTEN_KEEPALIVE
  BridgeSlotDescriptor* katago_get_slot_desc_ptr(int slot_id) {
    if(slot_id < 0 || slot_id >= KATAGO_MAX_SLOTS) {
      return NULL;
    }
    return &g_slot_descriptors[slot_id];
  }
}

struct LoadedModel {
  ModelDesc modelDesc;
  string modelFileName;

  LoadedModel(const string& fileName, const string& expectedSha256)
    : modelFileName(fileName)
  {
    (void)expectedSha256;
    // Check if fileName is JSON text or a JSON file
    bool isJsonText = (!fileName.empty() && fileName[0] == '{');
    bool isJsonFile = (fileName.size() >= 5 && fileName.substr(fileName.size() - 5) == ".json");

    if(isJsonText || isJsonFile) {
      json j;
      if(isJsonText) {
        j = json::parse(fileName);
      }
      else {
        ifstream ifs(fileName);
        if(!ifs.is_open()) {
          throw StringError("wasm_nnbackend: Could not open model json file: " + fileName);
        }
        ifs >> j;
      }

      modelDesc.name = j.value("name", "katago_model");
      modelDesc.sha256 = j.value("sha256", expectedSha256);
      modelDesc.modelVersion = j.value("modelVersion", 17);
      modelDesc.numInputChannels = j.value("numInputChannels", 22);
      modelDesc.numInputGlobalChannels = j.value("numInputGlobalChannels", 19);
      modelDesc.numInputMetaChannels = j.value("numInputMetaChannels", 0);
      modelDesc.numPolicyChannels = j.value("numPolicyChannels", 2);
      modelDesc.numValueChannels = j.value("numValueChannels", 3);
      modelDesc.numScoreValueChannels = j.value("numScoreValueChannels", 6);
      modelDesc.numOwnershipChannels = j.value("numOwnershipChannels", 1);
      modelDesc.metaEncoderVersion = j.value("metaEncoderVersion", 0);
      modelDesc.preferPassAliveUnderSuicideRules = j.value("preferPassAliveUnderSuicideRules", false);
      modelDesc.preferExcludeTerritoryAdjacentToAtari = j.value("preferExcludeTerritoryAdjacentToAtari", false);

      if(j.contains("postProcessParams")) {
        auto& p = j["postProcessParams"];
        modelDesc.postProcessParams.tdScoreMultiplier = p.value("tdScoreMultiplier", 20.0);
        modelDesc.postProcessParams.scoreMeanMultiplier = p.value("scoreMeanMultiplier", 20.0);
        modelDesc.postProcessParams.scoreStdevMultiplier = p.value("scoreStdevMultiplier", 20.0);
        modelDesc.postProcessParams.leadMultiplier = p.value("leadMultiplier", 20.0);
        modelDesc.postProcessParams.varianceTimeMultiplier = p.value("varianceTimeMultiplier", 40.0);
        modelDesc.postProcessParams.shorttermValueErrorMultiplier = p.value("shorttermValueErrorMultiplier", 0.25);
        modelDesc.postProcessParams.shorttermScoreErrorMultiplier = p.value("shorttermScoreErrorMultiplier", 150.0);
        modelDesc.postProcessParams.outputScaleMultiplier = p.value("outputScaleMultiplier", 1.0);
      }
      else {
        modelDesc.postProcessParams.tdScoreMultiplier = j.value("postProcess.tdScoreMultiplier", 20.0);
        modelDesc.postProcessParams.scoreMeanMultiplier = j.value("postProcess.scoreMeanMultiplier", 20.0);
        modelDesc.postProcessParams.scoreStdevMultiplier = j.value("postProcess.scoreStdevMultiplier", 20.0);
        modelDesc.postProcessParams.leadMultiplier = j.value("postProcess.leadMultiplier", 20.0);
        modelDesc.postProcessParams.varianceTimeMultiplier = j.value("postProcess.varianceTimeMultiplier", 40.0);
        modelDesc.postProcessParams.shorttermValueErrorMultiplier = j.value("postProcess.shorttermValueErrorMultiplier", 0.25);
        modelDesc.postProcessParams.shorttermScoreErrorMultiplier = j.value("postProcess.shorttermScoreErrorMultiplier", 150.0);
        modelDesc.postProcessParams.outputScaleMultiplier = j.value("postProcess.outputScaleMultiplier", 1.0);
      }
    }
    else {
      // If .onnx file or other, check for sidecar .json
      string jsonSidecar = fileName + ".json";
      ifstream ifs(jsonSidecar);
      if(ifs.is_open()) {
        json j;
        ifs >> j;
        modelDesc.name = j.value("name", "katago_model");
        modelDesc.sha256 = j.value("sha256", expectedSha256);
        modelDesc.modelVersion = j.value("modelVersion", 17);
        modelDesc.numInputChannels = j.value("numInputChannels", 22);
        modelDesc.numInputGlobalChannels = j.value("numInputGlobalChannels", 19);
        modelDesc.numInputMetaChannels = j.value("numInputMetaChannels", 0);
        modelDesc.numPolicyChannels = j.value("numPolicyChannels", 2);
        modelDesc.numValueChannels = j.value("numValueChannels", 3);
        modelDesc.numScoreValueChannels = j.value("numScoreValueChannels", 6);
        modelDesc.numOwnershipChannels = j.value("numOwnershipChannels", 1);
        modelDesc.metaEncoderVersion = j.value("metaEncoderVersion", 0);
        modelDesc.preferPassAliveUnderSuicideRules = j.value("preferPassAliveUnderSuicideRules", false);
        modelDesc.preferExcludeTerritoryAdjacentToAtari = j.value("preferExcludeTerritoryAdjacentToAtari", false);

        if(j.contains("postProcessParams")) {
          auto& p = j["postProcessParams"];
          modelDesc.postProcessParams.tdScoreMultiplier = p.value("tdScoreMultiplier", 20.0);
          modelDesc.postProcessParams.scoreMeanMultiplier = p.value("scoreMeanMultiplier", 20.0);
          modelDesc.postProcessParams.scoreStdevMultiplier = p.value("scoreStdevMultiplier", 20.0);
          modelDesc.postProcessParams.leadMultiplier = p.value("leadMultiplier", 20.0);
          modelDesc.postProcessParams.varianceTimeMultiplier = p.value("varianceTimeMultiplier", 40.0);
          modelDesc.postProcessParams.shorttermValueErrorMultiplier = p.value("shorttermValueErrorMultiplier", 0.25);
          modelDesc.postProcessParams.shorttermScoreErrorMultiplier = p.value("shorttermScoreErrorMultiplier", 150.0);
          modelDesc.postProcessParams.outputScaleMultiplier = p.value("outputScaleMultiplier", 1.0);
        }
      }
      else {
        // Fallback default for official b10c384 / transformer models
        modelDesc.name = "b10c384";
        modelDesc.sha256 = expectedSha256;
        modelDesc.modelVersion = 17;
        modelDesc.numInputChannels = 22;
        modelDesc.numInputGlobalChannels = 19;
        modelDesc.numInputMetaChannels = 0;
        modelDesc.numPolicyChannels = 2;
        modelDesc.numValueChannels = 3;
        modelDesc.numScoreValueChannels = 6;
        modelDesc.numOwnershipChannels = 1;
        modelDesc.metaEncoderVersion = 0;
        modelDesc.preferPassAliveUnderSuicideRules = false;
        modelDesc.preferExcludeTerritoryAdjacentToAtari = false;
        modelDesc.postProcessParams.tdScoreMultiplier = 20.0;
        modelDesc.postProcessParams.scoreMeanMultiplier = 20.0;
        modelDesc.postProcessParams.scoreStdevMultiplier = 20.0;
        modelDesc.postProcessParams.leadMultiplier = 20.0;
        modelDesc.postProcessParams.varianceTimeMultiplier = 40.0;
        modelDesc.postProcessParams.shorttermValueErrorMultiplier = 0.25;
        modelDesc.postProcessParams.shorttermScoreErrorMultiplier = 150.0;
        modelDesc.postProcessParams.outputScaleMultiplier = 1.0;
      }
    }
  }
};

struct ComputeContext {
  int nnXLen;
  int nnYLen;
  const LoadedModel* loadedModel;
};

struct ComputeHandle {
  ComputeContext* ctx;
  const LoadedModel* loadedModel;
  int maxBatchSize;
  bool requireExactNNLen;
  int slotId;
  int modelVersion;
  bool isWarmup;
};

struct InputBuffers {
  int maxBatchSize;
  size_t singleMaskElts;
  size_t singleInputElts;
  size_t singleInputGlobalElts;
  size_t singleInputMetaElts;

  size_t singlePolicyPassResultElts;
  size_t singlePolicyResultElts;
  size_t singleValueResultElts;
  size_t singleScoreValueResultElts;
  size_t singleOwnershipResultElts;

  vector<float> maskInput;
  vector<float> spatialInput;
  vector<float> globalInput;
  vector<float> metaInput;

  vector<float> policyPassOutput;
  vector<float> policyOutput;
  vector<float> valueOutput;
  vector<float> scoreValueOutput;
  vector<float> ownershipOutput;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    if(nnXLen > NNPos::MAX_BOARD_LEN) {
      throw StringError(Global::strprintf("nnXLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)", nnXLen, NNPos::MAX_BOARD_LEN));
    }
    if(nnYLen > NNPos::MAX_BOARD_LEN) {
      throw StringError(Global::strprintf("nnYLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)", nnYLen, NNPos::MAX_BOARD_LEN));
    }

    maxBatchSize = maxBatchSz;
    singleMaskElts = (size_t)nnXLen * nnYLen;
    singleInputElts = (size_t)m.numInputChannels * nnXLen * nnYLen;
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    singleInputMetaElts = (size_t)m.numInputMetaChannels;

    singlePolicyPassResultElts = (size_t)m.numPolicyChannels;
    singlePolicyResultElts = (size_t)m.numPolicyChannels * nnXLen * nnYLen;
    singleValueResultElts = (size_t)m.numValueChannels;
    singleScoreValueResultElts = (size_t)m.numScoreValueChannels;
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * nnXLen * nnYLen;

    maskInput.resize(singleMaskElts * maxBatchSize);
    spatialInput.resize(singleInputElts * maxBatchSize);
    globalInput.resize(singleInputGlobalElts * maxBatchSize);
    if(singleInputMetaElts > 0) {
      metaInput.resize(singleInputMetaElts * maxBatchSize);
    }

    policyPassOutput.resize(singlePolicyPassResultElts * maxBatchSize);
    policyOutput.resize(singlePolicyResultElts * maxBatchSize);
    valueOutput.resize(singleValueResultElts * maxBatchSize);
    scoreValueOutput.resize(singleScoreValueResultElts * maxBatchSize);
    ownershipOutput.resize(singleOwnershipResultElts * maxBatchSize);
  }
};

void NeuralNet::globalInitialize() {
  for(int i = 0; i < KATAGO_MAX_SLOTS; i++) {
    g_slot_descriptors[i].abi_version = KATAGO_BRIDGE_ABI_VERSION;
    g_slot_descriptors[i].slot_id = i;
    g_slot_descriptors[i].status = KATAGO_SLOT_FREE;
  }
}

void NeuralNet::globalCleanup() {
}

void NeuralNet::printDevices() {
  cout << "WebGPU / ONNX Runtime Web via JSPI Bridge" << endl;
}

std::string NeuralNet::getRuntimeBackendDetail(ConfigParser& cfg) {
  (void)cfg;
  return "webgpu-onnx-jspi";
}

NeuralNet::BatchPolicy NeuralNet::getBatchPolicy(ConfigParser& cfg) {
  (void)cfg;
  return NeuralNet::BatchPolicy::Dynamic;
}

int NeuralNet::getNumEffectiveDevices(ConfigParser& cfg, const std::vector<int>& gpuIdxByServerThread) {
  (void)cfg;
  std::set<int> distinctDevices(gpuIdxByServerThread.begin(), gpuIdxByServerThread.end());
  return std::max(1, (int)distinctDevices.size());
}

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  return new LoadedModel(file, expectedSha256);
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& homeDataDirOverride,
  enabled_t useFP16Mode,
  const LoadedModel* loadedModel,
  ConfigParser& cfg
) {
  (void)gpuIdxs;
  (void)logger;
  (void)homeDataDirOverride;
  (void)useFP16Mode;
  (void)cfg;

  ComputeContext* ctx = new ComputeContext();
  ctx->nnXLen = nnXLen;
  ctx->nnYLen = nnYLen;
  ctx->loadedModel = loadedModel;
  return ctx;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

static std::atomic<int> g_next_slot_id{0};

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  (void)logger;
  (void)inputsUseNHWC;
  (void)gpuIdxForThisThread;

  ComputeHandle* handle = new ComputeHandle();
  handle->ctx = context;
  handle->loadedModel = loadedModel;
  handle->maxBatchSize = maxBatchSize;
  handle->requireExactNNLen = requireExactNNLen;
  handle->slotId = (serverThreadIdx >= 0) ? (serverThreadIdx % KATAGO_MAX_SLOTS) : (g_next_slot_id.fetch_add(1) % KATAGO_MAX_SLOTS);
  handle->modelVersion = loadedModel->modelDesc.modelVersion;
  handle->isWarmup = false;
  return handle;
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  (void)handle;
  return false;
}

bool NeuralNet::setIsWarmup(const ComputeHandle* handle, bool isWarmup) {
  ComputeHandle* mutableHandle = const_cast<ComputeHandle*>(handle);
  bool old = mutableHandle->isWarmup;
  mutableHandle->isWarmup = isWarmup;
  return old;
}

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}

void NeuralNet::freeInputBuffers(InputBuffers* buffers) {
  delete buffers;
}

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);

  const int batchSize = numBatchEltsFilled;
  const int nnXLen = gpuHandle->ctx->nnXLen;
  const int nnYLen = gpuHandle->ctx->nnYLen;
  const int modelVersion = gpuHandle->modelVersion;

  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  const int numMetaFeatures = (int)inputBuffers->singleInputMetaElts;

  // 1. Fill input buffers
  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    float* rowMaskInput = inputBuffers->maskInput.data() + inputBuffers->singleMaskElts * nIdx;
    float* rowSpatialInput = inputBuffers->spatialInput.data() + inputBuffers->singleInputElts * nIdx;
    float* rowGlobalInput = inputBuffers->globalInput.data() + inputBuffers->singleInputGlobalElts * nIdx;

    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);
    SymmetryHelpers::copyInputsWithSymmetry(
      rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, false, inputBufs[nIdx]->symmetry);
    std::copy(rowSpatialInput, rowSpatialInput + inputBuffers->singleMaskElts, rowMaskInput);

    if(numMetaFeatures > 0) {
      float* rowMetaInput = inputBuffers->metaInput.data() + inputBuffers->singleInputMetaElts * nIdx;
      const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
      testAssert(inputBufs[nIdx]->hasRowMeta);
      std::copy(rowMeta, rowMeta + numMetaFeatures, rowMetaInput);
    }
  }

  // 2. Setup BridgeSlotDescriptor with dynamic slot lease
  uint64_t seq = 0;
  uint32_t gen = 0;
  int slotId = g_slotAllocator.acquireSlot(seq, gen);

  struct SlotGuard {
    int slot;
    ~SlotGuard() { g_slotAllocator.releaseSlot(slot); }
  } guard{slotId};

  BridgeSlotDescriptor& desc = g_slot_descriptors[slotId];

  desc.abi_version = KATAGO_BRIDGE_ABI_VERSION;
  desc.engine_generation = gen;
  desc.request_sequence = seq;
  desc.slot_id = slotId;
  desc.model_id = 0;
  desc.cancel_generation = 0;
  desc.error_code = KATAGO_BRIDGE_OK;
  desc.error_msg_offset = 0;
  desc.error_msg_capacity = 0;
  desc.reserved0 = 0;

  desc.batch_size = batchSize;
  desc.max_batch_size = inputBuffers->maxBatchSize;
  desc.nn_x_len = nnXLen;
  desc.nn_y_len = nnYLen;
  desc.num_spatial_channels = numSpatialFeatures;
  desc.num_global_channels = numGlobalFeatures;
  desc.num_meta_channels = numMetaFeatures;
  desc.num_policy_channels = (uint32_t)inputBuffers->singlePolicyPassResultElts;
  desc.num_value_channels = (uint32_t)inputBuffers->singleValueResultElts;
  desc.num_score_value_channels = (uint32_t)inputBuffers->singleScoreValueResultElts;
  desc.num_ownership_channels = (uint32_t)(inputBuffers->singleOwnershipResultElts / (nnXLen * nnYLen));
  desc.spatial_layout = KATAGO_LAYOUT_NCHW;

  desc.mask_offset = (uint32_t)(uintptr_t)inputBuffers->maskInput.data();
  desc.mask_bytes = (uint32_t)(inputBuffers->singleMaskElts * batchSize * sizeof(float));
  desc.spatial_offset = (uint32_t)(uintptr_t)inputBuffers->spatialInput.data();
  desc.spatial_bytes = (uint32_t)(inputBuffers->singleInputElts * batchSize * sizeof(float));
  desc.global_offset = (uint32_t)(uintptr_t)inputBuffers->globalInput.data();
  desc.global_bytes = (uint32_t)(inputBuffers->singleInputGlobalElts * batchSize * sizeof(float));
  desc.meta_offset = numMetaFeatures > 0 ? (uint32_t)(uintptr_t)inputBuffers->metaInput.data() : 0;
  desc.meta_bytes = (uint32_t)(inputBuffers->singleInputMetaElts * batchSize * sizeof(float));

  desc.policy_pass_offset = (uint32_t)(uintptr_t)inputBuffers->policyPassOutput.data();
  desc.policy_pass_bytes = (uint32_t)(inputBuffers->singlePolicyPassResultElts * batchSize * sizeof(float));
  desc.policy_offset = (uint32_t)(uintptr_t)inputBuffers->policyOutput.data();
  desc.policy_bytes = (uint32_t)(inputBuffers->singlePolicyResultElts * batchSize * sizeof(float));
  desc.value_offset = (uint32_t)(uintptr_t)inputBuffers->valueOutput.data();
  desc.value_bytes = (uint32_t)(inputBuffers->singleValueResultElts * batchSize * sizeof(float));
  desc.score_value_offset = (uint32_t)(uintptr_t)inputBuffers->scoreValueOutput.data();
  desc.score_value_bytes = (uint32_t)(inputBuffers->singleScoreValueResultElts * batchSize * sizeof(float));
  desc.ownership_offset = (uint32_t)(uintptr_t)inputBuffers->ownershipOutput.data();
  desc.ownership_bytes = (uint32_t)(inputBuffers->singleOwnershipResultElts * batchSize * sizeof(float));

  // Atomic state: READY
  __atomic_store_n(&desc.status, KATAGO_SLOT_READY, __ATOMIC_RELEASE);

  // 3. Suspend via JSPI to execute inference in coordinator worker
  int res = js_eval_batch_async(slotId, gen, (uint32_t)(seq & 0xFFFFFFFFULL), (uint32_t)(seq >> 32));
  if(res != 0) {
    __atomic_store_n(&desc.status, KATAGO_SLOT_ERROR, __ATOMIC_RELEASE);
    throw StringError("js_eval_batch_async returned error: " + Global::intToString(res));
  }

  // Verify status is DONE
  uint32_t curStatus = __atomic_load_n(&desc.status, __ATOMIC_ACQUIRE);
  if(curStatus != KATAGO_SLOT_DONE) {
    __atomic_store_n(&desc.status, KATAGO_SLOT_ERROR, __ATOMIC_RELEASE);
    throw StringError("Slot status was not KATAGO_SLOT_DONE after inference: " + Global::intToString(curStatus));
  }

  // 4. Decode outputs (matching upstream onnxbackend / trtbackend exactly)
  const float* policyPassData = inputBuffers->policyPassOutput.data();
  const float* policyData = inputBuffers->policyOutput.data();
  const float* valueData = inputBuffers->valueOutput.data();
  const float* scoreValueData = inputBuffers->scoreValueOutput.data();
  const float* ownershipData = inputBuffers->ownershipOutput.data();

  const int numPolicyChannels = (int)inputBuffers->singlePolicyPassResultElts;
  const int numValueChannels = (int)inputBuffers->singleValueResultElts;
  const int numScoreValueChannels = (int)inputBuffers->singleScoreValueResultElts;

  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);
    float policyOptimism = (float)inputBufs[row]->policyOptimism;

    // Policy decode
    {
      const float* policyPassSrcBuf = policyPassData + row * numPolicyChannels;
      const float* policySrcBuf = policyData + row * numPolicyChannels * nnXLen * nnYLen;
      float* policyProbs = output->policyProbs;

      if(numPolicyChannels == 2 || (numPolicyChannels == 4 && modelVersion >= 16)) {
        for(int i = 0; i < nnXLen * nnYLen; i++) {
          float p = policySrcBuf[i];
          float pOpt = policySrcBuf[i + nnXLen * nnYLen];
          policyProbsTmp[i] = p + (pOpt - p) * policyOptimism;
        }
        SymmetryHelpers::copyOutputsWithSymmetry(
          policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[nnXLen * nnYLen] =
          policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
      }
      else {
        assert(numPolicyChannels == 1);
        SymmetryHelpers::copyOutputsWithSymmetry(
          policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0];
      }
    }

    // Value decode
    {
      assert(numValueChannels == 3);
      output->whiteWinProb = valueData[row * numValueChannels];
      output->whiteLossProb = valueData[row * numValueChannels + 1];
      output->whiteNoResultProb = valueData[row * numValueChannels + 2];
    }

    // Ownership decode
    if(output->whiteOwnerMap != NULL) {
      assert(inputBuffers->singleOwnershipResultElts == (size_t)nnXLen * nnYLen);
      const float* ownershipSrcBuf = ownershipData + row * nnXLen * nnYLen;
      SymmetryHelpers::copyOutputsWithSymmetry(
        ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }

    // ScoreValue decode (ALL 6 channels preserved for modelVersion >= 9)
    {
      if(modelVersion >= 9) {
        assert(numScoreValueChannels == 6);
        output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
        output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
        output->whiteLead = scoreValueData[row * numScoreValueChannels + 2];
        output->varTimeLeft = scoreValueData[row * numScoreValueChannels + 3];
        output->shorttermWinlossError = scoreValueData[row * numScoreValueChannels + 4];
        output->shorttermScoreError = scoreValueData[row * numScoreValueChannels + 5];
      }
      else if(modelVersion >= 8) {
        assert(numScoreValueChannels == 4);
        output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
        output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
        output->whiteLead = scoreValueData[row * numScoreValueChannels + 2];
        output->varTimeLeft = scoreValueData[row * numScoreValueChannels + 3];
        output->shorttermWinlossError = 0;
        output->shorttermScoreError = 0;
      }
      else if(modelVersion >= 4) {
        assert(numScoreValueChannels == 2);
        output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
        output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
        output->whiteLead = output->whiteScoreMean;
        output->varTimeLeft = 0;
        output->shorttermWinlossError = 0;
        output->shorttermScoreError = 0;
      }
      else if(modelVersion >= 3) {
        assert(numScoreValueChannels == 1);
        output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
        output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
        output->whiteLead = output->whiteScoreMean;
        output->varTimeLeft = 0;
        output->shorttermWinlossError = 0;
        output->shorttermScoreError = 0;
      }
      else {
        ASSERT_UNREACHABLE;
      }
    }
  }

  // Reset slot status to FREE
  __atomic_store_n(&desc.status, KATAGO_SLOT_FREE, __ATOMIC_RELEASE);
}

// Stubs for testEvaluate functions
bool NeuralNet::testEvaluateConv(const ConvLayerDesc*, int, int, int, bool, bool, const std::vector<float>&, std::vector<float>&) {
  return false;
}
bool NeuralNet::testEvaluateBatchNorm(const BatchNormLayerDesc*, int, int, int, bool, bool, const std::vector<float>&, const std::vector<float>&, std::vector<float>&) {
  return false;
}
bool NeuralNet::testEvaluateResidualBlock(const ResidualBlockDesc*, int, int, int, bool, bool, const std::vector<float>&, const std::vector<float>&, std::vector<float>&) {
  return false;
}
bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(const GlobalPoolingResidualBlockDesc*, int, int, int, bool, bool, const std::vector<float>&, const std::vector<float>&, std::vector<float>&) {
  return false;
}
