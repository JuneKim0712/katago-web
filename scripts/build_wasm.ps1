# scripts/build_wasm.ps1
# Pinned Emscripten build for KataGo Web (Pattern C: Hybrid Emscripten C++ Wasm + WebGPU ONNX Bridge)

$ErrorActionPreference = "Stop"

$env:PATH = "C:\emsdk;C:\emsdk\upstream\emscripten;C:\Users\junwo\AppData\Local\Python\pythoncore-3.14-64\Scripts;$env:PATH"

$sources = @(
  "native/host/wasm_analysis_host.cpp",
  "native/backend/wasm_nnbackend.cpp",
  "native/upstream/cpp/core/global.cpp",
  "native/upstream/cpp/core/hash.cpp",
  "native/upstream/cpp/core/rand.cpp",
  "native/upstream/cpp/core/datetime.cpp",
  "native/upstream/cpp/core/fileutils.cpp",
  "native/upstream/cpp/core/makedir.cpp",
  "native/upstream/cpp/core/md5.cpp",
  "native/upstream/cpp/core/sha2.cpp",
  "native/upstream/cpp/core/test.cpp",
  "native/upstream/cpp/core/threadsafecounter.cpp",
  "native/upstream/cpp/core/threadsafequeue.cpp",
  "native/upstream/cpp/core/timer.cpp",
  "native/upstream/cpp/core/fancymath.cpp",
  "native/upstream/cpp/core/config_parser.cpp",
  "native/upstream/cpp/core/logger.cpp",
  "native/upstream/cpp/game/board.cpp",
  "native/upstream/cpp/game/rules.cpp",
  "native/upstream/cpp/game/boardhistory.cpp",
  "native/upstream/cpp/game/graphhash.cpp",
  "native/upstream/cpp/dataio/sgf.cpp",
  "native/upstream/cpp/neuralnet/nninputs.cpp",
  "native/upstream/cpp/neuralnet/sgfmetadata.cpp",
  "native/upstream/cpp/neuralnet/modelversion.cpp",
  "native/upstream/cpp/neuralnet/desc.cpp",
  "native/upstream/cpp/neuralnet/nneval.cpp",
  "native/upstream/cpp/search/analysisdata.cpp",
  "native/upstream/cpp/search/asyncbot.cpp",
  "native/upstream/cpp/search/distributiontable.cpp",
  "native/upstream/cpp/search/evalcache.cpp",
  "native/upstream/cpp/search/localpattern.cpp",
  "native/upstream/cpp/search/mutexpool.cpp",
  "native/upstream/cpp/search/patternbonustable.cpp",
  "native/upstream/cpp/search/reportedsearchvalues.cpp",
  "native/upstream/cpp/search/search.cpp",
  "native/upstream/cpp/search/searchexplorehelpers.cpp",
  "native/upstream/cpp/search/searchhelpers.cpp",
  "native/upstream/cpp/search/searchmirror.cpp",
  "native/upstream/cpp/search/searchmultithreadhelpers.cpp",
  "native/upstream/cpp/search/searchnnhelpers.cpp",
  "native/upstream/cpp/search/searchnode.cpp",
  "native/upstream/cpp/search/searchnodetable.cpp",
  "native/upstream/cpp/search/searchparams.cpp",
  "native/upstream/cpp/search/searchprint.cpp",
  "native/upstream/cpp/search/searchpuct.cpp",
  "native/upstream/cpp/search/searchresults.cpp",
  "native/upstream/cpp/search/searchtimehelpers.cpp",
  "native/upstream/cpp/search/searchupdatehelpers.cpp",
  "native/upstream/cpp/search/subtreevaluebiastable.cpp",
  "native/upstream/cpp/search/timecontrols.cpp",
  "native/upstream/cpp/program/playutils.cpp",
  "native/upstream/cpp/program/play.cpp",
  "native/upstream/cpp/program/playsettings.cpp",
  "native/upstream/cpp/program/setup.cpp"
)

$includes = @(
  "-I", "native/upstream/cpp",
  "-I", "native/upstream/cpp/external/filesystem-1.5.8/include",
  "-I", "native/upstream/cpp/external/tclap-1.2.5/include",
  "-I", "native/upstream/cpp/external/half-2.2.0/include",
  "-I", "native/upstream/cpp/external/nlohmann_json"
)

$exports = "_katago_init,_katago_load_model,_katago_post_analyze,_katago_poll_response,_katago_free_response,_katago_cancel,_katago_reset,_katago_destroy,_katago_get_slot_desc_ptr,_main,_malloc,_free"
$runtimeMethods = "ccall,cwrap,PThread,UTF8ToString,stringToUTF8,getValue,setValue"

$flags = @(
  "-O2",
  "-pthread",
  "-sPTHREAD_POOL_SIZE=8",
  "-sMALLOC=mimalloc",
  "-sJSPI=1",
  "-sJSPI_IMPORTS=js_eval_batch_async",
  "--js-library", "native/backend/library_nnbackend.js",
  "-fwasm-exceptions",
  "-sUSE_ZLIB=1",
  "-sSTACK_SIZE=8388608",
  "-sINITIAL_MEMORY=268435456",
  "-sALLOW_MEMORY_GROWTH=1",
  "-DNO_GIT_REVISION",
  "-sEXPORTED_FUNCTIONS=$exports",
  "-sEXPORTED_RUNTIME_METHODS=$runtimeMethods",
  "-sEXPORT_ES6=1",
  "-sMODULARIZE=1",
  "-o", "dist/wasm/katago.js"
)

Write-Host "Building KataGo Wasm engine with Emscripten..."
& emcc @sources @includes @flags
Write-Host "Build finished with exit code $LASTEXITCODE"
