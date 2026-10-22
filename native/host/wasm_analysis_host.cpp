#include "core/global.h"
#include "core/config_parser.h"
#include "core/timer.h"
#include "core/datetime.h"
#include "core/makedir.h"
#include "core/threadsafequeue.h"
#include "game/board.h"
#include "game/rules.h"
#include "game/boardhistory.h"
#include "search/asyncbot.h"
#include "search/patternbonustable.h"
#include "program/setup.h"
#include "program/playutils.h"
#include "program/play.h"
#include "neuralnet/nneval.h"
#include "neuralnet/nninterface.h"
#include "external/nlohmann_json/json.hpp"

#include <emscripten.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>

using namespace std;
using json = nlohmann::json;

static std::unique_ptr<NNEvaluator> g_nnEval;
static std::unique_ptr<AsyncBot> g_bot;
static SearchParams g_defaultParams;
static Logger g_logger;
static ThreadSafeQueue<string*> g_responseQueue;
static std::thread g_workerThread;
static std::atomic<bool> g_isAnalyzing{false};
static std::string g_currentRequestId;
static std::mutex g_botMutex;

extern "C" {

EMSCRIPTEN_KEEPALIVE
int katago_init() {
  try {
    Board::initHash();
    ScoreValue::initTables();
    NeuralNet::globalInitialize();

    g_defaultParams = SearchParams();
    g_defaultParams.numThreads = 2;
    g_defaultParams.maxVisits = 10000000;
    g_defaultParams.maxPlayouts = 10000000;

    return 0;
  }
  catch (const std::exception& e) {
    cerr << "katago_init error: " << e.what() << endl;
    return -1;
  }
}

EMSCRIPTEN_KEEPALIVE
int katago_load_model(
  const char* model_name,
  const char* model_json_or_file,
  int max_batch_size,
  int nn_x_len,
  int nn_y_len,
  int num_nn_threads
) {
  try {
    std::lock_guard<std::mutex> lock(g_botMutex);
    if(g_bot) {
      g_bot.reset();
    }
    if(g_nnEval) {
      g_nnEval.reset();
    }

    if(max_batch_size <= 0) max_batch_size = 4;
    if(nn_x_len <= 0) nn_x_len = 19;
    if(nn_y_len <= 0) nn_y_len = 19;
    if(num_nn_threads <= 0) num_nn_threads = 1;

    vector<int> gpuIdxByServerThread(num_nn_threads, 0);
    ConfigParser emptyCfg;

    g_nnEval = std::make_unique<NNEvaluator>(
      string(model_name ? model_name : "katago_model"),
      string(model_json_or_file ? model_json_or_file : ""),
      string(""), // expectedSha256
      &g_logger,
      max_batch_size,
      nn_x_len,
      nn_y_len,
      false, // requireExactNNLen
      false, // inputsUseNHWC
      -1,    // nnCacheSizePowerOfTwo
      -1,    // nnMutexPoolSizePowerOfTwo
      false, // debugSkipNeuralNet
      string(""), // homeDataDirOverride
      enabled_t::False, // useFP16Mode
      num_nn_threads,
      gpuIdxByServerThread,
      "randSeed123", // randSeed
      false, // doRandomize
      0,     // defaultSymmetry
      emptyCfg
    );

    g_nnEval->spawnServerThreads();

    string searchRandSeed = "searchRandSeed123";
    g_bot = std::make_unique<AsyncBot>(g_defaultParams, g_nnEval.get(), &g_logger, searchRandSeed);

    return 0;
  }
  catch (const std::exception& e) {
    cerr << "katago_load_model error: " << e.what() << endl;
    return -1;
  }
}

static void run_search_task(string query_str) {
  g_isAnalyzing.store(true, std::memory_order_release);
  try {
    json q = json::parse(query_str);
    string id = q.value("id", "req-0");
    g_currentRequestId = id;

    int boardXSize = q.value("boardXSize", 19);
    int boardYSize = q.value("boardYSize", 19);
    double komi = q.value("komi", 7.5);

    Rules rules;
    if(q.contains("rules")) {
      if(q["rules"].is_string()) {
        rules = Rules::parseRules(q["rules"].get<string>());
      }
      else {
        rules = Rules::getTrompTaylorish();
      }
    }
    else {
      rules = Rules::getTrompTaylorish();
    }
    rules.komi = (float)komi;

    if(q.contains("whiteHandicapBonus") && q["whiteHandicapBonus"].is_string()) {
      string whb = q["whiteHandicapBonus"].get<string>();
      try {
        rules.whiteHandicapBonusRule = Rules::parseWhiteHandicapBonusRule(whb);
      } catch(...) {}
    }

    Board board(boardXSize, boardYSize);
    Player nextPla = P_BLACK;
    if(q.contains("initialPlayer")) {
      string plaStr = q["initialPlayer"].get<string>();
      if(plaStr == "w" || plaStr == "W" || plaStr == "white" || plaStr == "WHITE") {
        nextPla = P_WHITE;
      }
    }

    // Initial stones
    if(q.contains("initialStones") && q["initialStones"].is_array()) {
      for(const auto& item : q["initialStones"]) {
        if(item.is_array() && item.size() >= 2) {
          string pStr = item[0].get<string>();
          Player p = (pStr == "w" || pStr == "W" || pStr == "white") ? P_WHITE : P_BLACK;
          string locStr = item[1].get<string>();
          Loc loc = Location::ofString(locStr, boardXSize, boardYSize);
          board.setStone(loc, p);
        }
      }
    }

    // Search params
    SearchParams params = g_defaultParams;
    if(q.contains("maxVisits")) params.maxVisits = q["maxVisits"].get<int64_t>();
    if(q.contains("maxPlayouts")) params.maxPlayouts = q["maxPlayouts"].get<int64_t>();
    if(q.contains("maxTime")) params.maxTime = q["maxTime"].get<double>();
    if(q.contains("numThreads")) params.numThreads = q["numThreads"].get<int>();
    if(q.contains("rootNumSymmetriesToSample")) params.rootNumSymmetriesToSample = q["rootNumSymmetriesToSample"].get<int>();
    if(q.contains("conservativePass")) params.conservativePass = q["conservativePass"].get<bool>();

    // Nested search object support
    if(q.contains("search") && q["search"].is_object()) {
      auto& s = q["search"];
      if(s.contains("maxVisits")) params.maxVisits = s["maxVisits"].get<int64_t>();
      if(s.contains("maxPlayouts")) params.maxPlayouts = s["maxPlayouts"].get<int64_t>();
      if(s.contains("maxTimeMs")) params.maxTime = s["maxTimeMs"].get<double>() / 1000.0;
      if(s.contains("numThreads")) params.numThreads = s["numThreads"].get<int>();
      if(s.contains("rootNumSymmetriesToSample")) params.rootNumSymmetriesToSample = s["rootNumSymmetriesToSample"].get<int>();
      if(s.contains("conservativePass")) params.conservativePass = s["conservativePass"].get<bool>();
    }

    // Override settings (wideRootNoise, searchRandSeed)
    if(q.contains("overrideSettings") && q["overrideSettings"].is_object()) {
      auto& o = q["overrideSettings"];
      if(o.contains("wideRootNoise")) params.wideRootNoise = o["wideRootNoise"].get<double>();
    }

    // Allow / Avoid moves restrictions
    std::vector<int> avoidMoveUntilByLocBlack(Board::MAX_ARR_SIZE, 0);
    std::vector<int> avoidMoveUntilByLocWhite(Board::MAX_ARR_SIZE, 0);
    bool hasAvoidRestrictions = false;

    if(q.contains("allowMoves") && q["allowMoves"].is_array()) {
      for(const auto& item : q["allowMoves"]) {
        if(item.contains("player") && item.contains("moves") && item["moves"].is_array()) {
          string pStr = item["player"].get<string>();
          Player p = (pStr == "w" || pStr == "W" || pStr == "white") ? P_WHITE : P_BLACK;
          int untilDepth = item.value("untilDepth", 1);
          auto& avoidVec = (p == P_BLACK) ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;
          std::fill(avoidVec.begin(), avoidVec.end(), untilDepth);
          for(const auto& m : item["moves"]) {
            string locStr = m.get<string>();
            Loc loc = Location::ofString(locStr, boardXSize, boardYSize);
            avoidVec[loc] = 0;
          }
          hasAvoidRestrictions = true;
        }
      }
    }

    BoardHistoryModes historyModes = Search::resolveHistoryModes(params, g_nnEval.get());
    BoardHistory hist(board, nextPla, rules, 0, historyModes);

    // Moves replay
    if(q.contains("moves") && q["moves"].is_array()) {
      for(const auto& item : q["moves"]) {
        if(item.is_array() && item.size() >= 2) {
          string pStr = item[0].get<string>();
          Player p = (pStr == "w" || pStr == "W" || pStr == "white") ? P_WHITE : P_BLACK;
          string locStr = item[1].get<string>();
          Loc loc = Location::ofString(locStr, boardXSize, boardYSize);
          if(p != nextPla) {
            board.clearSimpleKoLoc();
            hist.clear(board, p, rules, hist.encorePhase);
          }
          bool suc = hist.makeBoardMoveTolerant(board, loc, p, false);
          if(!suc) {
            json err;
            err["error"] = "Illegal move in history: " + locStr;
            err["id"] = id;
            g_responseQueue.forcePush(new string(err.dump()));
            g_isAnalyzing.store(false, std::memory_order_release);
            return;
          }
          nextPla = getOpp(p);
        }
      }
    }

    // Output options
    int analysisPVLen = q.value("analysisPVLen", 10);
    bool includeOwnership = q.value("includeOwnership", true);
    bool includeOwnershipStdev = q.value("includeOwnershipStdev", false);
    bool includePolicy = q.value("includePolicy", true);
    double reportDuringSearchEvery = q.value("reportDuringSearchEvery", 0.0);

    if(q.contains("output") && q["output"].is_object()) {
      auto& o = q["output"];
      if(o.contains("pvLength")) analysisPVLen = o["pvLength"].get<int>();
      if(o.contains("includeOwnership")) includeOwnership = o["includeOwnership"].get<bool>();
      if(o.contains("includeOwnershipStdev")) includeOwnershipStdev = o["includeOwnershipStdev"].get<bool>();
      if(o.contains("includePolicy")) includePolicy = o["includePolicy"].get<bool>();
      if(o.contains("progressIntervalMs")) {
        double ms = o["progressIntervalMs"].get<double>();
        if(ms > 0) reportDuringSearchEvery = ms / 1000.0;
      }
    }

    // Configure bot under mutex lock, then release before searching
    {
      std::lock_guard<std::mutex> lock(g_botMutex);
      if(!g_bot) {
        json err;
        err["error"] = "Bot not initialized";
        err["id"] = id;
        g_responseQueue.forcePush(new string(err.dump()));
        g_isAnalyzing.store(false, std::memory_order_release);
        return;
      }

      g_bot->setPosition(nextPla, board, hist);
      g_bot->setAlwaysIncludeOwnerMap(includeOwnership || includeOwnershipStdev);
      g_bot->setParams(params);
      if(hasAvoidRestrictions) {
        g_bot->setAvoidMoveUntilByLoc(avoidMoveUntilByLocBlack, avoidMoveUntilByLocWhite);
      }
    }

    Player pla = nextPla;
    double searchFactor = 1.0;

    auto reportSnapshot = [id, analysisPVLen, includePolicy, includeOwnership, includeOwnershipStdev](const Search* search, bool isDuringSearch) {
      json ret;
      ret["id"] = id;
      ret["isDuringSearch"] = isDuringSearch;
      bool success = search->getAnalysisJson(
        P_BLACK, // Canonical BLACK perspective
        analysisPVLen,
        false, // preventEncore
        includePolicy,
        includeOwnership,
        includeOwnershipStdev,
        false, // includeMovesOwnership
        false, // includeMovesOwnershipStdev
        false, // includePVVisits
        true,  // STAGE R5 FIX: includeNoResultValue enabled
        ret
      );
      if(success) {
        g_responseQueue.forcePush(new string(ret.dump()));
      }
      return success;
    };

    // Run search WITHOUT holding g_botMutex so katago_cancel can execute stopWithoutWait non-blockingly
    if(reportDuringSearchEvery > 0.0) {
      std::function<void(const Search* search)> callback = [&reportSnapshot](const Search* search) {
        reportSnapshot(search, true);
      };
      std::function<void()> onSearchBegun = []() {};

      g_bot->genMoveSynchronousAnalyze(
        pla, TimeControls(), searchFactor,
        reportDuringSearchEvery, 0.05,
        callback, onSearchBegun
      );
    }
    else {
      std::function<void()> onSearchBegun = []() {};
      g_bot->genMoveSynchronous(pla, TimeControls(), searchFactor, onSearchBegun);
    }

    // Final result
    const Search* search = g_bot->getSearch();
    reportSnapshot(search, false);
    g_bot->clearSearch();
  }
  catch (const std::exception& e) {
    json err;
    err["error"] = string("Exception during analysis: ") + e.what();
    err["id"] = g_currentRequestId;
    g_responseQueue.forcePush(new string(err.dump()));
  }
  g_isAnalyzing.store(false, std::memory_order_release);
}

EMSCRIPTEN_KEEPALIVE
int katago_post_analyze(const char* query_json) {
  if(!query_json) return -1;
  if(g_isAnalyzing.load(std::memory_order_acquire)) {
    return -2; // BUSY
  }

  if(g_workerThread.joinable()) {
    g_workerThread.join();
  }

  string query_str(query_json);
  g_workerThread = std::thread(run_search_task, query_str);
  return 0;
}

EMSCRIPTEN_KEEPALIVE
char* katago_poll_response() {
  string* msg = nullptr;
  if(g_responseQueue.tryPop(msg)) {
    char* cstr = (char*)malloc(msg->size() + 1);
    memcpy(cstr, msg->c_str(), msg->size() + 1);
    delete msg;
    return cstr;
  }
  return nullptr;
}

EMSCRIPTEN_KEEPALIVE
void katago_free_response(char* ptr) {
  if(ptr) free(ptr);
}

EMSCRIPTEN_KEEPALIVE
int katago_cancel(const char* request_id) {
  std::lock_guard<std::mutex> lock(g_botMutex);
  if(g_bot) {
    if(request_id && strlen(request_id) > 0) {
      if(g_currentRequestId != request_id) {
        return 0; // Stale or mismatched request ID, ignore
      }
    }
    g_bot->stopWithoutWait();
    return 0;
  }
  return -1;
}

EMSCRIPTEN_KEEPALIVE
int katago_reset() {
  {
    std::lock_guard<std::mutex> lock(g_botMutex);
    if(g_bot) {
      g_bot->stopWithoutWait();
    }
  }
  if(g_workerThread.joinable()) {
    g_workerThread.join();
  }
  std::lock_guard<std::mutex> lock(g_botMutex);
  g_bot.reset();
  g_nnEval.reset();
  g_isAnalyzing.store(false, std::memory_order_release);

  string* msg = nullptr;
  while(g_responseQueue.tryPop(msg)) {
    delete msg;
  }
  return 0;
}

EMSCRIPTEN_KEEPALIVE
int katago_destroy() {
  katago_reset();
  NeuralNet::globalCleanup();
  return 0;
}

int main() {
  return 0;
}

} // extern "C"
