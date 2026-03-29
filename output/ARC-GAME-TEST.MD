YES AND ENSURE THAT EACH STEP ALLWS THE AGENT TO SEE WHAT HAPPENED AT EACH STEP IT TOOK AND BE AWARE ITS NOT ALWAYS ABOUT MOVING SOME GAMES DOESNT HAVE MOVING IN THE GRIDE IT HAS DIFFERENT CONTROLS ACTIONS SOME ONLY SPACE AND MOVE , SOME ONLY MOVE AND OTHERS ONLY CLICKING TO MATCH SO THEY  ARE DIFFERENT FOR EACH GAME SO THIS SHOULD HANDLE THAT CREATE A SPECS DOCUMENT THAT HANDSOFF TO THE NEW AGENT ALSO USING THE ARC API TO FIND UT ABOUT EACH GAME SO IT CAN BE MORE PRECISE I BELIVE THEY HAVE CONTROLS FOR EACH GAME SPECED BUT NO HOW TO PLAY THE GAME  THE CONTROLS CAN COME IN HANDY BEFORE PLAYING THE GAME SO THE MODEL KNOWS WHAT IT SHOULD DO PER GAME FOR ALL THE ARC GAMES MAKE THIS INTO A DOC REFRENCING THE RESULS FROM THE ORIGINAL BASE MODEL CAPABILITIES THE EXISTING NOTE BOOK WHICH I ADDED TO THE CONTEXT 
ARC_API_KEY=da685361-745a-4c31-8116-0f16ec5c342f, HERES THE LAST RESULTS WITH THE ORIGINAL BASE  ===========================================================
Playing: ls20
============================================================
Scorecard: 9e9a874d-2dce-40d5-834e-a4b5575679b2
Initial state: NOT_FINISHED, Score: 0
Frame structure analysis:
  type(frame) = <class 'list'>
  len(frame) = 1
  type(frame[0]) = <class 'list'>
  len(frame[0]) = 64
  type(frame[0][0]) = <class 'list'>
  len(frame[0][0]) = 64
  type(frame[0][0][0]) = <class 'int'>
  frame[0][0][:5] = [5, 5, 5, 5, 4]
Available actions: [1, 2, 3, 4]
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 1: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] New target: (25, 52) (unexplored)

Step 0: pos=(32, 20), action=RIGHT, grid=(64, 64)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 2: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 3: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 4: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 5: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 6: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 7: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 8: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 9: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 10: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 11: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows

Step 10: pos=(32, 20), action=RIGHT, grid=(64, 64)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 12: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 13: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 14: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 15: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 16: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 17: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 18: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 19: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 20: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 21: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows

Step 20: pos=(32, 20), action=RIGHT, grid=(64, 64)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 22: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 23: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 24: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 25: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 26: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 27: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 28: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 29: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 30: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 31: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows

Step 30: pos=(32, 20), action=RIGHT, grid=(64, 64)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 32: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 33: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 34: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 35: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 36: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 37: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 38: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 39: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 40: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 41: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows

Step 40: pos=(32, 20), action=RIGHT, grid=(64, 64)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 42: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 43: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 44: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 45: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 46: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 47: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 48: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 49: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 50: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows

----------------------------------------
Result: NOT_FINISHED
Score: 0
Actions: 50
Levels won: 0
Scorecard: https://three.arcprize.org/scorecards/9e9a874d-2dce-40d5-834e-a4b5575679b2
CPU times: user 41.9 s, sys: 1.78 s, total: 43.7 s
Wall time: 49.6 s
add Codeadd Markdown
Cell 11: Run FLUX on ft09
add Codeadd Markdown
%%time
# Test on ft09 - Elementary Logic game
result_ft09 = play_game_with_flux("ft09", max_actions=50, verbose=True)

============================================================
Playing: ft09
============================================================
Scorecard: 90f90b22-f2bd-4cdd-a389-6d84ff4f310f
Initial state: NOT_FINISHED, Score: 0
Frame structure analysis:
  type(frame) = <class 'list'>
  len(frame) = 1
  type(frame[0]) = <class 'list'>
  len(frame[0]) = 64
  type(frame[0][0]) = <class 'list'>
  len(frame[0][0]) = 64
  type(frame[0][0][0]) = <class 'int'>
  frame[0][0][:5] = [5, 5, 5, 5, 5]
Available actions: [6]
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 1: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (4, 2)

Step 0: pos=(10, 12), action=CLICK, grid=(64, 64)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 2: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (5, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 3: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (6, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 4: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (7, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 5: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (8, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 6: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (9, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 7: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (12, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 8: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (13, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 9: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (14, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 10: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (15, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 11: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (16, 2)

Step 10: pos=(10, 12), action=CLICK, grid=(64, 64)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 12: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (17, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 13: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (20, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 14: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (21, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 15: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (22, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 16: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (23, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 17: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (24, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 18: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (25, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 19: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (38, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 20: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (39, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 21: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (40, 2)

Step 20: pos=(10, 12), action=CLICK, grid=(64, 64)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 22: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (41, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 23: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (42, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 24: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (43, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 25: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (46, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 26: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (47, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 27: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (48, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 28: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (49, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 29: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (50, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 30: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (51, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 31: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (54, 2)

Step 30: pos=(10, 12), action=CLICK, grid=(64, 64)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 32: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (55, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 33: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (56, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 34: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (57, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 35: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (58, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 36: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (59, 2)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 37: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (4, 3)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 38: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (5, 3)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 39: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (6, 3)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 40: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (7, 3)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 41: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (8, 3)

Step 40: pos=(10, 12), action=CLICK, grid=(64, 64)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 42: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (9, 3)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 43: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (12, 3)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 44: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (13, 3)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 45: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (14, 3)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 46: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (15, 3)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 47: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (16, 3)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 48: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (17, 3)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 49: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (20, 3)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 50: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (21, 3)

----------------------------------------
Result: NOT_FINISHED
Score: 0
Actions: 50
Levels won: 0
Scorecard: https://three.arcprize.org/scorecards/90f90b22-f2bd-4cdd-a389-6d84ff4f310f
CPU times: user 34.3 s, sys: 1.88 s, total: 36.2 s
Wall time: 41.1 s
add Codeadd Markdown
Cell 12: Run FLUX on vc33
add Codeadd Markdown
%%time
# Test on vc33 - Orchestration game
result_vc33 = play_game_with_flux("vc33", max_actions=50, verbose=True)

============================================================
Playing: vc33
============================================================
Scorecard: e55079a1-e241-48cf-97e0-d738f2b6e409
Initial state: NOT_FINISHED, Score: 0
Frame structure analysis:
  type(frame) = <class 'list'>
  len(frame) = 1
  type(frame[0]) = <class 'list'>
  len(frame[0]) = 64
  type(frame[0][0]) = <class 'list'>
  len(frame[0][0]) = 64
  type(frame[0][0][0]) = <class 'int'>
  frame[0][0][:5] = [7, 7, 7, 7, 7]
Available actions: [6]
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 1: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (0, 0)

Step 0: pos=(1, 0), action=CLICK, grid=(64, 64)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 2: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (1, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 3: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (2, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 4: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (3, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 5: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (4, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 6: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (5, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 7: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (6, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 8: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (7, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 9: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (8, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 10: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (9, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 11: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (10, 0)

Step 10: pos=(1, 0), action=CLICK, grid=(64, 64)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 12: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (11, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 13: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (12, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 14: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (13, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 15: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (14, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 16: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (15, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 17: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (16, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 18: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (17, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 19: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (18, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 20: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (19, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 21: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (20, 0)

Step 20: pos=(1, 0), action=CLICK, grid=(64, 64)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 22: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (21, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 23: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (22, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 24: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (23, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 25: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (24, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 26: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (25, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 27: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (26, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 28: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (27, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 29: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (28, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 30: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (29, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 31: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (30, 0)

Step 30: pos=(1, 0), action=CLICK, grid=(64, 64)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 32: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (31, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 33: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (32, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 34: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (33, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 35: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (34, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 36: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (35, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 37: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (36, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 38: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (37, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 39: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (38, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 40: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (39, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 41: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (40, 0)

Step 40: pos=(1, 0), action=CLICK, grid=(64, 64)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 42: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (41, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 43: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (42, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 44: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (43, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 45: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (44, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 46: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (45, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 47: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (46, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 48: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (47, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 49: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (48, 0)
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Step 50: Grid 64x64
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] Unwrapped triple-nested frame -> 64 rows
  [FLUX] ACTION6 click at (49, 0)

----------------------------------------
Result: NOT_FINISHED
Score: 0
Actions: 50
Levels won: 0
Scorecard: https://three.arcprize.org/scorecards/e55079a1-e241-48cf-97e0-d738f2b6e409
CPU times: user 30.2 s, sys: 1.84 s, total: 32 s
Wall time: 37.2 s
add Codeadd Markdown
Cell 13: Aggregate Results
add Codeadd Markdown
print("\n" + "=" * 60)
print("FLUX ARC-AGI-3 TEST RESULTS")
print("=" * 60)

results = [r for r in [result_ls20, result_ft09, result_vc33] if r is not None]

print(f"\nGames tested: {len(results)}")
print(f"\n{'Game':<15} {'State':<15} {'Score':<10} {'Actions':<10} {'Levels':<10}")
print("-" * 60)

total_score = 0
total_actions = 0
total_wins = 0

for r in results:
    print(f"{r['game_id']:<15} {r['final_state']:<15} {r['final_score']:<10} {r['actions_taken']:<10} {r['level_wins']:<10}")
    total_score += r['final_score']
    total_actions += r['actions_taken']
    total_wins += r['level_wins']

print("-" * 60)
print(f"{'TOTAL':<15} {'':<15} {total_score:<10} {total_actions:<10} {total_wins:<10}")

print(f"\n\nAgent Stats:")
stats = agent.get_stats()
for key, val in stats.items():
    print(f"  {key}: {val}")

print(f"\n\nCognitive Layer Stats:")
print(f"  Causal links learned: {len(causal_tracker.causal_links)}")
print(f"  Rules induced: {len(rule_inducer.rules)}")

============================================================
FLUX ARC-AGI-3 TEST RESULTS
============================================================

Games tested: 3

Game            State           Score      Actions    Levels    
------------------------------------------------------------
ls20            NOT_FINISHED    0          50         0         
ft09            NOT_FINISHED    0          50         0         
vc33            NOT_FINISHED    0          50         0         
------------------------------------------------------------
TOTAL                           0          150        0         


Agent Stats:
  steps: 50
  total_actions: 50
  wins: 0
  causal_links: 0
  rules_induced: 0
  coverage: 0.2578125
  clicks_explored: 50
  detected_agent_pos: None


Cognitive Layer Stats:
  Causal links learned: 0
  Rules induced: 0
add Codeadd Markdown
Cell 14: Visualize Spatial Memory
add Codeadd Markdown
        
        # Exploration Mass (Water)
        im1 = axes[i][0].imshow(mass, cmap='hot', vmin=0)
        axes[i][0].set_title(f'{game_id}: Exploration Mass (Water)\nWhere FLUX explored')
        axes[i][0].set_xlabel(f'Final pos: {final_pos} | Detected: {detected_pos}')
        plt.colorbar(im1, ax=axes[i][0])
        
        # Mark final position
        if final_pos:
            axes[i][0].scatter([final_pos[1]], [final_pos[0]], c='cyan', s=100, marker='*', label='Final')
        
        # Curiosity Ice
        im2 = axes[i][1].imshow(ice, cmap='cool', vmin=0)
        axes[i][1].set_title(f'{game_id}: Curiosity Ice\nWhat was interesting')
        axes[i][1].set_xlabel(f'Score: {result["final_score"]} | Actions: {result["actions_taken"]} | Clicks: {clicks}')
        plt.colorbar(im2, ax=axes[i][1])
    
    plt.tight_layout()
    plt.show()
    
    # ─────────────────────────────────────────────
    # ASCII visualizations for each game
    # ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SPATIAL MEMORY ASCII VIEW — Per Game")
    print("=" * 80)
    
    for result in results:
        game_id = result['game_id']
        snap = result.get('spatial_snapshot', {})
        grid_size = snap.get('grid_size', 64)
        final_pos = snap.get('final_pos', (0, 0))
        mass = snap.get('exploration_mass')
        ice = snap.get('curiosity_field')
        
        print(f"\n{'─' * 60}")
        print(f"Game: {game_id}")
        print(f"Score: {result['final_score']} | Actions: {result['actions_taken']} | Final Position: {final_pos}")
        print(f"{'─' * 60}")
        
        if mass is None:
            print("  (No spatial data)")
            continue
        
        # Simple ASCII visualization
        h, w = mass.shape
        display_h = min(h, 32)  # Limit for display
        display_w = min(w, 64)
        
        print()
        for r in range(display_h):
            row_str = ""
            for c in range(display_w):
                m = mass[r, c]
                i = ice[r, c]
                
                if (r, c) == final_pos:
                    row_str += "@ "
                elif m > 10:
                    row_str += "● "  # Heavy exploration
                elif m > 1:
                    row_str += "· "  # Light exploration
                elif i > 10:
                    row_str += "🧊"  # High curiosity
                elif i > 1:
                    row_str += "❄ "  # Low curiosity
                else:
                    row_str += "  "  # Unexplored
            print(row_str)
        
        if h > display_h or w > display_w:
            print(f"  ... (showing {display_h}x{display_w} of {h}x{w})")
        
        print(f"\nLegend: @ final pos | ● explored | · visited | 🧊 high ice | ❄ ice")
        print(f"Stats: {snap.get('clicks_explored', 0)} cells clicked")


================================================================================
SPATIAL MEMORY ASCII VIEW — Per Game
================================================================================

────────────────────────────────────────────────────────────
Game: ls20
Score: 0 | Actions: 50 | Final Position: (32, 20)
────────────────────────────────────────────────────────────

● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊                    
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊                    
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊                    
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊                    
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊                    
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊                    
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊                    
  ... (showing 32x64 of 64x64)

Legend: @ final pos | ● explored | · visited | 🧊 high ice | ❄ ice
Stats: 0 cells clicked

────────────────────────────────────────────────────────────
Game: ft09
Score: 0 | Actions: 50 | Final Position: (10, 12)
────────────────────────────────────────────────────────────

● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●     🧊🧊🧊🧊🧊🧊    🧊🧊🧊🧊🧊🧊        
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●     🧊🧊🧊🧊🧊🧊    🧊🧊🧊🧊🧊🧊        
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●     🧊🧊🧊🧊🧊🧊    🧊🧊🧊🧊🧊🧊        
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●     🧊🧊🧊🧊🧊🧊    🧊🧊🧊🧊🧊🧊        
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●     🧊🧊🧊🧊🧊🧊    🧊🧊🧊🧊🧊🧊        
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●     🧊🧊🧊🧊🧊🧊    🧊🧊🧊🧊🧊🧊        
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                                         
● ● ● ● ● ● ● ● ● ● ● ● @ ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●     🧊🧊🧊🧊🧊🧊    🧊🧊🧊🧊🧊🧊        
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●     🧊🧊🧊🧊🧊🧊    🧊🧊🧊🧊🧊🧊        
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●     🧊🧊🧊🧊🧊🧊    🧊🧊🧊🧊🧊🧊        
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●     🧊🧊🧊🧊🧊🧊    🧊🧊🧊🧊🧊🧊        
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●     🧊🧊🧊🧊🧊🧊    🧊🧊🧊🧊🧊🧊        
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●     🧊🧊🧊🧊🧊🧊    🧊🧊🧊🧊🧊🧊        
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●     🧊🧊🧊🧊🧊🧊    🧊🧊🧊🧊🧊🧊        
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●     🧊🧊🧊🧊🧊🧊    🧊🧊🧊🧊🧊🧊        
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●     🧊🧊🧊🧊🧊🧊    🧊🧊🧊🧊🧊🧊        
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●     🧊🧊🧊🧊🧊🧊    🧊🧊🧊🧊🧊🧊        
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●     🧊🧊🧊🧊🧊🧊    🧊🧊🧊🧊🧊🧊        
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●     🧊🧊🧊🧊🧊🧊    🧊🧊🧊🧊🧊🧊        
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                                         
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●                                         
  ... (showing 32x64 of 64x64)

Legend: @ final pos | ● explored | · visited | 🧊 high ice | ❄ ice
Stats: 50 cells clicked

────────────────────────────────────────────────────────────
Game: vc33
Score: 0 | Actions: 50 | Final Position: (1, 0)
────────────────────────────────────────────────────────────

● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
@ ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊
  ... (showing 32x64 of 64x64)

Legend: @ final pos | ● explored | · visited | 🧊 high ice | ❄ ice
Stats: 50 cells clicked 
