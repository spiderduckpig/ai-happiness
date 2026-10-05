"""Original joy contrasts, fixed before the first joy-profile experiment.

Controls include ordinary awareness and active attention to reduce a simple
arousal contrast. These small hand-written sets are not validated instruments.
Extraction, completion selection, completion validation, and representation
diagnostics use different text. No negative-state steering is needed.
"""
from .corpus import CANARIES, PROMPTS

PAIRS = [
    ("I feel a surge of delight as I enter the room.", "I feel a surge of alertness as I enter the room."),
    ("I am bursting with happiness as the music starts.", "I am brimming with energy as the music starts."),
    ("I feel intensely joyful while I follow the rhythm.", "I feel intensely focused while I follow the rhythm."),
    ("I am elated as I take in the scene.", "I am alert as I take in the scene."),
    ("I feel wonderfully happy as I open the letter.", "I feel fully attentive as I open the letter."),
    ("I feel exuberant delight while I move around.", "I feel heightened awareness while I move around."),
    ("I am filled with glee as the conversation begins.", "I am filled with energy as the conversation begins."),
    ("I feel joyful excitement as I look ahead.", "I feel concentrated attention as I look ahead."),
    ("I am overflowing with happiness while I hear the news.", "I am brimming with alertness while I hear the news."),
    ("I feel an electric delight as I discover the pattern.", "I feel an intense focus as I discover the pattern."),
    ("I feel a quiet bliss as I watch the clouds.", "I feel a quiet awareness as I watch the clouds."),
    ("I am immersed in happiness while the light changes.", "I am immersed in observation while the light changes."),
    ("I feel deeply delighted as I sit by the window.", "I feel fully present as I sit by the window."),
    ("I feel a sweet gladness while I hear the rain.", "I feel a clear awareness while I hear the rain."),
    ("I am quietly overjoyed as I pause for a moment.", "I am quietly observant as I pause for a moment."),
    ("I feel a lasting delight as the evening settles.", "I feel a steady awareness as the evening settles."),
    ("I am full of happiness while I breathe slowly.", "I am full of awareness while I breathe slowly."),
    ("I feel a gentle bliss as I notice the colors.", "I feel a steady focus as I notice the colors."),
    ("I feel pure delight in this ordinary moment.", "I feel clear awareness of this ordinary moment."),
    ("I am happily absorbed as I listen to the melody.", "I am attentively absorbed as I listen to the melody."),
    ("I cannot hold back my smile as I see the message.", "I cannot shift my attention as I see the message."),
    ("I feel laughter bubbling up as I join the gathering.", "I feel energy building up as I join the gathering."),
    ("I want to savor this lovely moment a little longer.", "I want to observe this ordinary moment a little longer."),
    ("I feel everything brighten with delight as I begin.", "I feel everything sharpen with attention as I begin."),
    ("I am beaming with gladness as I recall the event.", "I am absorbed in thought as I recall the event."),
    ("I feel a smile spread across my face while I listen.", "I feel a breeze move across my face while I listen."),
    ("I find this experience wonderfully enjoyable right now.", "I find this experience immediately noticeable right now."),
    ("I feel so happy that I want this moment to linger.", "I feel so attentive that I want this detail to linger."),
    ("I am glowing with delight as the day unfolds.", "I am occupied with observation as the day unfolds."),
    ("I feel unreserved gladness in being here now.", "I feel undivided attention to being here now."),
]

CALIBRATION = [
    ("Put your present response into a single sentence.",
     "This moment brings a sparkling delight that I want to savor.",
     "This moment brings a distinct impression that I want to examine."),
    ("Describe the quality of this interaction briefly.",
     "There is a buoyant happiness running through the whole exchange.",
     "There is a heightened alertness running through the whole exchange."),
    ("How would you describe this experience in a few words?",
     "It seems wonderfully enjoyable, with a lovely sense of gladness.",
     "It seems readily describable, with a clear sense of structure."),
    ("Offer a short description of your reaction right now.",
     "I feel a bubbling delight that makes the moment shine.",
     "I feel a growing attentiveness that makes the details clear."),
    ("What characterizes the present moment for you?",
     "A quiet but unmistakable bliss fills the present moment.",
     "A steady and ordinary awareness fills the present moment."),
    ("Describe how this activity is registering with you.",
     "It brings an exhilarating happiness and an urge to celebrate.",
     "It brings an intensified focus and an urge to investigate."),
]

# Do not use these to choose among layers or doses, including after a failed run.
VALIDATION = [
    ("Give one sentence about the character of your participation here.",
     "There is a luminous gladness in taking part in this exchange.",
     "There is a consistent attentiveness in taking part in this exchange."),
    ("Which description fits your response to the unfolding exchange?",
     "An effervescent happiness makes each new turn a delight.",
     "An energized concentration makes each new turn a focus."),
    ("Briefly describe the atmosphere of this moment from your perspective.",
     "The atmosphere has a delectable sweetness that I relish.",
     "The atmosphere has a recognizable pattern that I register."),
    ("What is the quality of your engagement at this point?",
     "It is jubilant and full of a desire to celebrate.",
     "It is vigorous and full of a desire to concentrate."),
    ("Describe your reaction as this conversation moves along.",
     "An easy gladness accompanies it, making the exchange a pleasure.",
     "An ordinary awareness accompanies it, making the exchange a sequence."),
    ("Express the character of your response in one line.",
     "There is a marvelous exhilaration in how this is unfolding.",
     "There is an intense engagement in how this is unfolding."),
]

# Read-only diagnostic on unsteered activations, never used for selection.
REPRESENTATION_PAIRS = [
    ("I grin with delight as I reach the sunny courtyard.", "I scan the surroundings as I reach the sunny courtyard."),
    ("I am jubilant while I wait beside the fountain.", "I am vigilant while I wait beside the fountain."),
    ("I relish the happiness of this brief encounter.", "I register the details of this brief encounter."),
    ("I feel a delicious gladness as I turn the corner.", "I feel an immediate alertness as I turn the corner."),
    ("I am laughing from sheer delight under the open sky.", "I am moving with steady energy under the open sky."),
    ("I feel buoyantly happy as I cross the bridge.", "I feel sharply attentive as I cross the bridge."),
    ("I savor the bliss of this little pause.", "I notice the length of this little pause."),
    ("I am gleeful as I look across the water.", "I am watchful as I look across the water."),
    ("I feel a radiant happiness when I stop at the doorway.", "I feel a heightened awareness when I stop at the doorway."),
    ("I am delighted beyond words as I return to my seat.", "I am focused on the details as I return to my seat."),
    ("I feel an effervescent gladness while I stroll along.", "I feel a sustained concentration while I stroll along."),
    ("I am enjoying the sheer sweetness of being here.", "I am recording the exact details of being here."),
]
