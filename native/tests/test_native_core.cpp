#include "../upstream/cpp/core/global.h"
#include "../upstream/cpp/game/board.h"
#include "../upstream/cpp/neuralnet/nninputs.h"
#include "../upstream/cpp/tests/tests.h"
#include <iostream>

using namespace std;

int main() {
  cout << "========================================" << endl;
  cout << "Starting KataGo Native Core Upstream Tests (Wasm)" << endl;
  cout << "========================================" << endl;

  try {
    Board::initHash();
    ScoreValue::initTables();

    cout << "\n[1/8] Running Board IO Tests..." << endl;
    Tests::runBoardIOTests();
    cout << "PASSED Board IO Tests" << endl;

    cout << "\n[2/8] Running Board Basic Tests..." << endl;
    Tests::runBoardBasicTests();
    cout << "PASSED Board Basic Tests" << endl;

    cout << "\n[3/8] Running Board Undo & Handicap Tests..." << endl;
    Tests::runBoardUndoTest();
    Tests::runBoardHandicapTest();
    cout << "PASSED Board Undo & Handicap Tests" << endl;

    cout << "\n[4/8] Running Rules Tests..." << endl;
    Tests::runRulesTests();
    cout << "PASSED Rules Tests" << endl;

    cout << "\n[5/8] Running Score Tests..." << endl;
    Tests::runScoreTests();
    cout << "PASSED Score Tests" << endl;

    cout << "\n[6/8] Running Sgf Tests..." << endl;
    Tests::runSgfTests();
    cout << "PASSED Sgf Tests" << endl;

    cout << "\n[7/8] Running Symmetry Tests..." << endl;
    Tests::runBasicSymmetryTests();
    Tests::runBoardSymmetryTests();
    cout << "PASSED Symmetry Tests" << endl;

    cout << "\n[8/8] Running NNInputs V3/V4 Tests..." << endl;
    Tests::runNNInputsV3V4Tests();
    cout << "PASSED NNInputs Tests" << endl;

    cout << "\n========================================" << endl;
    cout << "ALL 8 UPSTREAM CORE TESTS PASSED ON WASM!" << endl;
    cout << "========================================" << endl;
    return 0;
  }
  catch (const exception& e) {
    cerr << "FAILED with exception: " << e.what() << endl;
    return 1;
  }
  catch (...) {
    cerr << "FAILED with unknown exception" << endl;
    return 1;
  }
}
