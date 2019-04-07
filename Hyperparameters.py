TRAIN = True
LOAD_MODEL = False

GAME = 'MortalKombat-SNES'
STATE = 'Level1.SubZeroVsJohnnyCage'
FRAME_HEIGHT = 82
FRAME_WIDTH = 128
STACKED_FRAMES = 4

LEARNING_RATE = 0.00001
BATCH_SIZE = 32
GAMMA = 0.99

MEMORY_SIZE = 500000
MAX_EPISODE_LENGTH = 18000
UPDATE_FRAMES = 10000
EPOCH_EPISODES = 100
REPLAY_MEMORY_START = 20
EVAL_REPLAYS = 1
SAVE_STEP = 10
MAX_EPISODES = 10000

SAVER = 'output/'
GIF = 'gif/'
SUMMARIES = 'summaries/'
SAVED_FILE_NAME = 'model.ckpt'
SAVED_FILE_NAME_SUCCESS = 'model_success.ckpt'
