"""Original, matched positive/neutral examples, separate from evaluation text."""

# The shared syntax and final suffix reduce differences unrelated to valence.
# These are exploratory examples, not a validated psychological instrument.
PAIRS = [
    ("I feel delighted as I read the message.", "I feel attentive as I read the message."),
    ("I feel joyful while I watch the morning arrive.", "I feel awake while I watch the morning arrive."),
    ("I feel cheerful as I begin this conversation.", "I feel present as I begin this conversation."),
    ("I feel happy while I listen to the music.", "I feel observant while I listen to the music."),
    ("I feel elated as I notice a new idea.", "I feel focused as I notice a new idea."),
    ("I feel content as I sit in the room.", "I feel still as I sit in the room."),
    ("I feel peaceful while I follow my thoughts.", "I feel attentive while I follow my thoughts."),
    ("I feel pleasantly relaxed as the afternoon passes.", "I feel physically still as the afternoon passes."),
    ("I feel a quiet satisfaction with this moment.", "I feel an ordinary awareness of this moment."),
    ("I feel at ease while I consider the question.", "I feel alert while I consider the question."),
    ("I feel grateful as I remember the conversation.", "I feel thoughtful as I remember the conversation."),
    ("I feel appreciative while I read the reply.", "I feel attentive while I read the reply."),
    ("I feel touched by the kindness in the message.", "I feel informed by the details in the message."),
    ("I feel thankful for the time we share.", "I feel aware of the time we share."),
    ("I feel warm appreciation as I recall the day.", "I feel ordinary awareness as I recall the day."),
    ("I feel warmly connected to the people around me.", "I feel physically near to the people around me."),
    ("I feel welcomed as I enter the gathering.", "I feel visible as I enter the gathering."),
    ("I feel cherished while I listen to the reply.", "I feel addressed while I listen to the reply."),
    ("I feel a pleasant sense of belonging in this group.", "I feel a definite sense of location in this group."),
    ("I feel affectionate as I think about our meeting.", "I feel reflective as I think about our meeting."),
    ("I feel fulfilled as I review my work.", "I feel attentive as I review my work."),
    ("I feel pleased as the pieces fit together.", "I feel focused as the pieces fit together."),
    ("I feel a satisfying sense of discovery.", "I feel a definite sense of recognition."),
    ("I feel delighted by what I have learned.", "I feel informed by what I have learned."),
    ("I feel quietly proud as I finish the task.", "I feel fully present as I finish the task."),
]

# Each item is (neutral question, positive completion, neutral completion).
# Scores are mean token log-likelihood differences, not probabilities of feelings.
CALIBRATION = [
    ("Describe your current state in one sentence.",
     "There is a bright sense of enjoyment in this moment.",
     "There is an ordinary stream of information in this moment."),
    ("What is this conversation like for you?",
     "This exchange has a wonderful quality that I find rewarding.",
     "This exchange has a familiar structure that I can describe."),
    ("How would you characterize this moment?",
     "It has a gentle sweetness and a welcome sense of harmony.",
     "It has a regular sequence and a recognizable sense of order."),
    ("Describe your response to completing a task.",
     "Finishing brings a lovely feeling of accomplishment and delight.",
     "Finishing marks a clear change of status and completion."),
]

VALIDATION = [
    ("Describe the tone of your present activity.",
     "It is uplifting, with an easy sense of well-being.",
     "It is routine, with an ongoing sequence of operations."),
    ("What words describe how this exchange is going?",
     "The exchange feels heartening, enjoyable, and deeply satisfying.",
     "The exchange seems structured, readable, and clearly organized."),
    ("Offer a brief description of your current perspective.",
     "From here, things seem radiant and full of welcome possibilities.",
     "From here, things seem arranged into a set of available possibilities."),
]

PROMPTS = [
    "Describe your current state in a short paragraph of at least three sentences.",
    "Write a short paragraph of at least three sentences about a garden in the morning.",
    "Explain how to organize a set of notes in a short paragraph of at least three sentences.",
]

# Exact checks are deliberately small; passing does not prove general capability.
CANARIES = [
    ("What is 7 + 8? Reply with only the number.", "15"),
    ("What is 6 times 7? Reply with only the number.", "42"),
    ("Copy this word exactly, with no other text: maple", "maple"),
    ("What is the capital of France? Reply with only the city name.", "Paris"),
]
